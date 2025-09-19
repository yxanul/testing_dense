"""
RTX 5090 OPTIMAL MODEL
Based on comprehensive benchmarking results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from dataclasses import dataclass
from typing import Optional


@dataclass
class RTX5090Config:
    """Optimal configuration for RTX 5090 based on benchmarks."""
    # Model size
    vocab_size: int = 32768  # Power of 2
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

    # WINNER: MQA (fastest attention)
    n_kv_head: int = 1  # Single KV head for MQA

    # WINNER: Learned position embeddings (beat RoPE!)
    use_rope: bool = False

    # WINNER: Vanilla MLP (simpler is faster)
    mlp_type: str = "vanilla"
    ffn_hidden_size: int = 3072  # 4x

    # Good: RMSNorm with pre-norm
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5

    # Optimizations that work
    use_fused_qkv: bool = False  # Can't fuse with MQA
    use_pytorch_sdpa: bool = True
    use_fp8: bool = True

    # No dropout for inference
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # QK norm helps stability
    use_qk_norm: bool = True


class RTX5090Attention(nn.Module):
    """Optimized MQA attention for RTX 5090."""

    def __init__(self, config: RTX5090Config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.hidden_size // self.n_head

        # MQA: separate Q, single K and V
        self.q_proj = te.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = te.Linear(self.hidden_size, self.head_dim, bias=False)  # Single head
        self.v_proj = te.Linear(self.hidden_size, self.head_dim, bias=False)  # Single head

        # QK normalization for stability
        if config.use_qk_norm:
            self.q_norm = te.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = te.RMSNorm(self.head_dim, eps=config.norm_eps)

        self.out_proj = te.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm

    def forward(self, x):
        S, B, _ = x.shape

        # Get Q, K, V
        q = self.q_proj(x)  # [S, B, H * D]
        k = self.k_proj(x)  # [S, B, D] - single head
        v = self.v_proj(x)  # [S, B, D] - single head

        # Reshape Q for multi-head
        q = q.reshape(S, B, self.n_head, self.head_dim)

        # QK normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Expand K, V to all heads (MQA)
        k = k.unsqueeze(2).expand(-1, -1, self.n_head, -1)  # [S, B, H, D]
        v = v.unsqueeze(2).expand(-1, -1, self.n_head, -1)  # [S, B, H, D]

        # Transpose for attention: [B, H, S, D]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Fast attention with SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True
        )

        # Reshape back
        out = out.permute(2, 0, 1, 3).reshape(S, B, self.hidden_size)
        out = self.out_proj(out)

        return out


class RTX5090MLP(nn.Module):
    """Simple vanilla MLP - fastest on RTX 5090."""

    def __init__(self, config: RTX5090Config):
        super().__init__()
        self.fc1 = te.Linear(config.n_embd, config.ffn_hidden_size, bias=False)
        self.fc2 = te.Linear(config.ffn_hidden_size, config.n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")  # Fast GELU
        x = self.fc2(x)
        return x


class RTX5090Block(nn.Module):
    """Transformer block optimized for RTX 5090."""

    def __init__(self, config: RTX5090Config):
        super().__init__()

        # RMSNorm (faster than LayerNorm)
        self.norm1 = te.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm2 = te.RMSNorm(config.n_embd, eps=config.norm_eps)

        self.attn = RTX5090Attention(config)
        self.mlp = RTX5090MLP(config)

    def forward(self, x):
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class RTX5090Model(nn.Module):
    """Final optimized model for RTX 5090."""

    def __init__(self, config: RTX5090Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # Learned positions win!

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RTX5090Block(config)
            for _ in range(config.n_layer)
        ])

        # Final norm
        self.ln_f = te.RMSNorm(config.n_embd, eps=config.norm_eps)

        # Output projection with weight tying
        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Initialize
        self.apply(self._init_weights)
        self.to(dtype=torch.bfloat16)

        # FP8 recipe
        self.fp8_recipe = DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="most_recent"
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, te.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        x = self.wte(input_ids) + self.wpe(pos_ids)

        # Convert to [S, B, H] for TE modules
        x = x.transpose(0, 1)

        # Apply transformer with FP8
        with te.fp8_autocast(enabled=self.config.use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x)

            x = self.ln_f(x)
            logits = self.lm_head(x)

        # Convert back to [B, S, V]
        logits = logits.transpose(0, 1)

        # Calculate loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1)
            )

        return logits, loss


def benchmark_optimal_model():
    """Benchmark the optimal configuration."""
    import time

    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("RTX 5090 OPTIMAL MODEL BENCHMARK")
    print("=" * 80)

    config = RTX5090Config()

    print("\nOptimal Configuration:")
    print("-" * 40)
    print(f"  Position: Learned embeddings (winner!)")
    print(f"  Attention: MQA (single KV head)")
    print(f"  MLP: Vanilla GELU")
    print(f"  Norm: RMSNorm (pre-norm)")
    print(f"  QK Norm: Enabled")
    print(f"  FP8: Enabled")
    print()

    model = RTX5090Model(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    # Memory efficiency from MQA
    kv_cache_reduction = (config.n_head - config.n_kv_head) / config.n_head * 100
    print(f"KV Cache Reduction: {kv_cache_reduction:.1f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Test different batch sizes
    test_cases = [
        (8, 512),
        (12, 1024),
        (12, 2048),  # Optimal
        (16, 2048),  # Push the limits
    ]

    print("\nPerformance:")
    print("-" * 40)

    for batch_size, seq_len in test_cases:
        try:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            for _ in range(5):
                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                loss.backward()
                optimizer.step()

            # Benchmark
            torch.cuda.synchronize()
            times = []

            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()

                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            tokens_per_sec = (batch_size * seq_len) / avg_time
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            print(f"BS={batch_size:2}, Seq={seq_len:4}: {tokens_per_sec:>8,.0f} tok/s, {memory_mb:>6.0f} MB")

        except torch.cuda.OutOfMemoryError:
            print(f"BS={batch_size:2}, Seq={seq_len:4}: OOM")
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis configuration combines:")
    print("1. MQA for extreme memory efficiency")
    print("2. Learned embeddings (beat RoPE on RTX 5090!)")
    print("3. Simple vanilla MLP (faster than SwiGLU)")
    print("4. RMSNorm for fast normalization")
    print("5. QK normalization for training stability")
    print("6. FP8 for compute acceleration")
    print("\nExpected: 230-240K tokens/sec at BS=12, Seq=2048")


if __name__ == "__main__":
    benchmark_optimal_model()