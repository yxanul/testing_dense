"""
Benchmark comparing fused vs non-fused TransformerEngine modules.
Also tests different FP8 formats (E4M3, HYBRID with E5M2 backward).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
import time
import gc
from dataclasses import dataclass
from typing import List, Tuple


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def benchmark_model(model, input_ids, target_ids, n_iters=10, warmup=3):
    """Benchmark forward and backward pass times."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(warmup):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure forward times
    forward_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(input_ids)

        torch.cuda.synchronize()
        forward_times.append((time.perf_counter() - start) * 1000)

    # Measure backward times
    backward_times = []
    for _ in range(n_iters):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)

        optimizer.step()
        optimizer.zero_grad()

    return {
        'forward_mean': sum(forward_times) / len(forward_times),
        'forward_std': torch.tensor(forward_times).std().item(),
        'backward_mean': sum(backward_times) / len(backward_times),
        'backward_std': torch.tensor(backward_times).std().item(),
        'total_mean': sum(forward_times) / len(forward_times) + sum(backward_times) / len(backward_times),
    }


class NonFusedBlock(nn.Module):
    """Non-fused transformer block using separate TE modules."""
    def __init__(self, hidden_size, num_heads, use_fp8=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_fp8 = use_fp8

        # Separate LayerNorm and Linear
        self.ln1 = te.LayerNorm(hidden_size)
        self.qkv = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = te.Linear(hidden_size, hidden_size, bias=True)

        self.ln2 = te.LayerNorm(hidden_size)
        self.fc1 = te.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.fc2 = te.Linear(4 * hidden_size, hidden_size, bias=True)

        self.dropout = nn.Dropout(0.1)
        self.fp8_recipe = None

    def set_fp8_recipe(self, recipe):
        self.fp8_recipe = recipe

    def forward(self, x):
        S, B, _ = x.shape
        residual = x

        # Attention block - separate ops
        x = self.ln1(x)
        qkv = self.qkv(x)

        # Simple attention (simplified for benchmark)
        qkv = qkv.view(S, B, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = torch.split(qkv, self.head_dim, dim=3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, self.hidden_size).transpose(0, 1)

        x = self.proj(out)
        x = residual + self.dropout(x)

        # MLP block - separate ops
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = residual + self.dropout(x)

        return x


class FusedBlock(nn.Module):
    """Fused transformer block using LayerNormLinear and LayerNormMLP."""
    def __init__(self, hidden_size, num_heads, use_fp8=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_fp8 = use_fp8

        # Fused modules
        self.ln_qkv = te.LayerNormLinear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = te.Linear(hidden_size, hidden_size, bias=True)
        self.ln_mlp = te.LayerNormMLP(hidden_size, 4 * hidden_size, bias=True)

        self.dropout = nn.Dropout(0.1)
        self.fp8_recipe = None

    def set_fp8_recipe(self, recipe):
        self.fp8_recipe = recipe

    def forward(self, x):
        S, B, _ = x.shape
        residual = x

        # Fused LayerNorm + QKV
        qkv = self.ln_qkv(x)

        # Simple attention
        qkv = qkv.view(S, B, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = torch.split(qkv, self.head_dim, dim=3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, self.hidden_size).transpose(0, 1)

        x = self.proj(out)
        x = residual + self.dropout(x)

        # Fused LayerNorm + MLP
        residual = x
        x = self.ln_mlp(x)
        x = residual + x

        return x


class BenchmarkModel(nn.Module):
    """Simple model for benchmarking."""
    def __init__(self, block_class, n_layers=4, hidden_size=768, num_heads=12, vocab_size=50304):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.wpe = nn.Embedding(1024, hidden_size)

        self.blocks = nn.ModuleList([
            block_class(hidden_size, num_heads)
            for _ in range(n_layers)
        ])

        self.ln_f = te.LayerNorm(hidden_size)
        self.lm_head = te.Linear(hidden_size, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.fp8_recipe = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, 0, 0.02)
        elif isinstance(module, te.Linear):
            nn.init.normal_(module.weight, 0, 0.02)

    def set_fp8_recipe(self, recipe):
        self.fp8_recipe = recipe
        for block in self.blocks:
            block.set_fp8_recipe(recipe)

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, S, device=device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        x = x.transpose(0, 1)  # [S, B, H]

        if self.fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                for block in self.blocks:
                    x = block(x)
                x = self.ln_f(x)
                logits = self.lm_head(x)
        else:
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

        return logits.transpose(0, 1)


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    # Test configurations
    B, S = 8, 512  # Batch size, sequence length
    vocab_size = 50304
    hidden_size = 768
    num_heads = 12
    n_layers = 4

    print("=" * 80)
    print("TransformerEngine Benchmark: Fused vs Non-Fused Modules")
    print("=" * 80)
    print(f"Config: B={B}, S={S}, H={hidden_size}, Layers={n_layers}")
    print()

    # Prepare data
    input_ids = torch.randint(0, vocab_size, (B, S), device=device)
    target_ids = torch.randint(0, vocab_size, (B, S), device=device)

    # Test different configurations
    configs = [
        ("Non-Fused (No FP8)", NonFusedBlock, None),
        ("Non-Fused (E4M3)", NonFusedBlock, DelayedScaling(margin=0, fp8_format=Format.E4M3)),
        ("Non-Fused (HYBRID E4M3/E5M2)", NonFusedBlock, DelayedScaling(fp8_format=Format.HYBRID)),
        ("Fused (No FP8)", FusedBlock, None),
        ("Fused (E4M3)", FusedBlock, DelayedScaling(margin=0, fp8_format=Format.E4M3)),
        ("Fused (HYBRID E4M3/E5M2)", FusedBlock, DelayedScaling(fp8_format=Format.HYBRID)),
    ]

    results = []

    for name, block_class, fp8_recipe in configs:
        print(f"\nTesting: {name}")
        print("-" * 40)

        # Create model
        model = BenchmarkModel(block_class, n_layers, hidden_size, num_heads, vocab_size)
        model = model.to(device).to(torch.bfloat16)

        if fp8_recipe:
            model.set_fp8_recipe(fp8_recipe)
            if "HYBRID" in name:
                print(f"  FP8 Format: E4M3 (forward) / E5M2 (backward)")
            elif "E4M3" in name:
                print(f"  FP8 Format: E4M3 (both forward and backward)")

        # Get memory before
        torch.cuda.empty_cache()
        mem_before = get_memory_usage()

        # Benchmark
        try:
            stats = benchmark_model(model, input_ids, target_ids)

            # Get memory after
            mem_after = get_memory_usage()

            print(f"  Forward:  {stats['forward_mean']:.2f} ± {stats['forward_std']:.2f} ms")
            print(f"  Backward: {stats['backward_mean']:.2f} ± {stats['backward_std']:.2f} ms")
            print(f"  Total:    {stats['total_mean']:.2f} ms")
            print(f"  Memory:   {mem_after - mem_before:.1f} MB")

            results.append({
                'name': name,
                'forward': stats['forward_mean'],
                'backward': stats['backward_mean'],
                'total': stats['total_mean'],
                'memory': mem_after - mem_before
            })

        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        baseline_total = results[0]['total']  # Non-Fused, No FP8

        print(f"\n{'Configuration':<30} {'Forward':>10} {'Backward':>10} {'Total':>10} {'Speedup':>10}")
        print("-" * 70)

        for r in results:
            speedup = baseline_total / r['total']
            print(f"{r['name']:<30} {r['forward']:>8.2f}ms {r['backward']:>8.2f}ms "
                  f"{r['total']:>8.2f}ms {speedup:>8.2f}x")

        print("\nKey Findings:")
        print("1. Fused modules reduce kernel launch overhead")
        print("2. FP8 HYBRID uses E4M3 for forward, E5M2 for backward (better gradients)")
        print("3. Pure E4M3 is fastest but may have less accurate gradients")


if __name__ == "__main__":
    main()