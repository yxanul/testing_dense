"""
Alternative implementation using Local/Block Attention
Instead of sliding windows, use block-local attention which is more efficient.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from dataclasses import dataclass
from typing import Optional
from model_te_final_v2 import FinalConfig, FinalMLP, FinalTransformerBlock, FinalGPT2Model


class LocalBlockAttention(nn.Module):
    """Local block attention - each position attends to a fixed block around it.

    Much more efficient than sliding window as it can be implemented with
    tensor reshaping rather than masking.
    """

    def __init__(self, config: FinalConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.hidden_size // self.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.layer_idx = layer_idx

        # Local attention configuration
        self.block_size = getattr(config, 'block_size', 256)  # Size of attention blocks
        self.use_local_attn = (
            getattr(config, 'use_local_attention', False) and
            getattr(config, 'local_attention_layers', None) and
            layer_idx in config.local_attention_layers
        )

        # Fused QKV projection
        if config.use_fused_qkv:
            total_proj_size = (
                self.hidden_size +  # Q
                self.n_kv_head * self.head_dim +  # K
                self.n_kv_head * self.head_dim  # V
            )
            self.qkv_proj = te.Linear(self.hidden_size, total_proj_size, bias=True)
        else:
            self.q_proj = te.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.k_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=True)
            self.v_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=True)

        self.out_proj = te.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.use_sdpa = config.use_pytorch_sdpa

    def forward(self, x):
        S, B, _ = x.shape

        # Get QKV
        if hasattr(self, 'qkv_proj'):
            qkv = self.qkv_proj(x)
            q_size = self.hidden_size
            kv_size = self.n_kv_head * self.head_dim
            q = qkv[..., :q_size]
            k = qkv[..., q_size:q_size + kv_size]
            v = qkv[..., q_size + kv_size:]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Reshape for attention
        q = q.reshape(S, B, self.n_head, self.head_dim)
        k = k.reshape(S, B, self.n_kv_head, self.head_dim)
        v = v.reshape(S, B, self.n_kv_head, self.head_dim)

        # GQA: repeat KV heads if needed
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        if self.use_local_attn and S > self.block_size:
            # Use block-local attention
            out = self._local_block_attention(q, k, v, S, B)
        else:
            # Standard full attention
            q = q.permute(1, 2, 0, 3)  # [B, H, S, D]
            k = k.permute(1, 2, 0, 3)
            v = v.permute(1, 2, 0, 3)

            if self.use_sdpa:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))
                attn = F.softmax(scores, dim=-1)
                attn = self.attn_dropout(attn)
                out = torch.matmul(attn, v)

            out = out.permute(2, 0, 1, 3).reshape(S, B, self.hidden_size)

        out = self.out_proj(out)
        return out

    def _local_block_attention(self, q, k, v, S, B):
        """Efficient block-local attention using tensor reshaping.

        Instead of creating masks, we reshape the sequence into blocks
        and compute attention within each block independently.
        """
        # Calculate number of blocks
        n_blocks = (S + self.block_size - 1) // self.block_size

        # Pad sequence to be divisible by block_size
        pad_len = n_blocks * self.block_size - S
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad_len))

        # Reshape into blocks: [n_blocks, block_size, B, H, D]
        q = q.reshape(n_blocks, self.block_size, B, self.n_head, self.head_dim)
        k = k.reshape(n_blocks, self.block_size, B, self.n_head, self.head_dim)
        v = v.reshape(n_blocks, self.block_size, B, self.n_head, self.head_dim)

        # Permute for batch processing: [n_blocks * B, H, block_size, D]
        q = q.permute(0, 2, 3, 1, 4).reshape(-1, self.n_head, self.block_size, self.head_dim)
        k = k.permute(0, 2, 3, 1, 4).reshape(-1, self.n_head, self.block_size, self.head_dim)
        v = v.permute(0, 2, 3, 1, 4).reshape(-1, self.n_head, self.block_size, self.head_dim)

        # Compute attention within each block
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True  # Still causal within blocks
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Causal mask for blocks
            block_mask = torch.triu(torch.ones(self.block_size, self.block_size, device=q.device), diagonal=1).bool()
            scores.masked_fill_(block_mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)

        # Reshape back: [n_blocks, B, H, block_size, D]
        out = out.reshape(n_blocks, B, self.n_head, self.block_size, self.head_dim)

        # Permute back: [n_blocks * block_size, B, H, D]
        out = out.permute(0, 3, 1, 2, 4).reshape(-1, B, self.hidden_size)

        # Remove padding if necessary
        if pad_len > 0:
            out = out[:S]

        return out


class LocalTransformerBlock(nn.Module):
    """Transformer block with local attention option."""

    def __init__(self, config: FinalConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # Normalization
        if config.use_rmsnorm:
            self.norm1 = te.RMSNorm(config.n_embd, eps=config.layernorm_epsilon)
            self.norm2 = te.RMSNorm(config.n_embd, eps=config.layernorm_epsilon)
        else:
            self.norm1 = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)
            self.norm2 = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)

        # Use local attention
        self.attn = LocalBlockAttention(config, layer_idx)

        # MLP without fusion
        self.mlp = FinalMLP(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + self.dropout(x)

        # Pre-norm MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class LocalGPT2Model(FinalGPT2Model):
    """GPT-2 with local block attention for efficiency."""

    def __init__(self, config: FinalConfig):
        # Initialize parent class first
        super().__init__(config)

        # Replace transformer blocks with local attention blocks if configured
        if getattr(config, 'use_local_attention', False):
            self.blocks = nn.ModuleList([
                LocalTransformerBlock(config, layer_idx=i)
                for i in range(config.n_layer)
            ])


def get_local_attention_config():
    """Config with local block attention for efficiency."""
    config = FinalConfig(
        vocab_size=32768,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=3,  # GQA 4:1
        mlp_type="swiglu",
        use_fused_qkv=True,
        use_fused_mlp=False,
        use_rmsnorm=True,
        use_pytorch_sdpa=True,
        use_fp8=True,
    )
    # Add local attention configuration
    config.use_local_attention = True
    config.block_size = 256  # Each block attends to 256 tokens
    config.local_attention_layers = [1, 3, 5, 7, 9, 11]  # Odd layers use local attention
    return config


def benchmark_local_attention():
    """Benchmark local block attention vs full attention."""
    import time

    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("LOCAL BLOCK ATTENTION BENCHMARK")
    print("=" * 80)

    # Standard config
    standard_config = FinalConfig(
        vocab_size=32768,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=3,
        mlp_type="swiglu",
        use_fused_qkv=True,
        use_fused_mlp=False,
        use_rmsnorm=True,
        use_fp8=True,
    )

    # Local attention config
    local_config = get_local_attention_config()

    test_cases = [
        (8, 512),
        (8, 1024),
        (8, 2048),
        (4, 4096),
    ]

    print("\nComparing Full Attention vs Local Block Attention:")
    print("-" * 60)

    for batch_size, seq_len in test_cases:
        print(f"\nBS={batch_size}, Seq={seq_len}:")

        results = {}

        for name, config in [("Full", standard_config), ("Local", local_config)]:
            try:
                model = LocalGPT2Model(config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
                labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

                # Warmup
                for _ in range(5):
                    optimizer.zero_grad()
                    with te.fp8_autocast(enabled=config.use_fp8):
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
                    with te.fp8_autocast(enabled=config.use_fp8):
                        logits, loss = model(input_ids, labels=labels)
                    loss.backward()
                    optimizer.step()

                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

                avg_time = sum(times) / len(times)
                tokens_per_sec = (batch_size * seq_len) / avg_time
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

                results[name] = {
                    'tokens_per_sec': tokens_per_sec,
                    'memory_mb': peak_memory,
                    'ms_per_iter': avg_time * 1000
                }

                print(f"  {name:6}: {tokens_per_sec:>8,.0f} tok/s, {peak_memory:>6.0f} MB, {avg_time*1000:>6.1f} ms")

                del model
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  {name:6}: OOM")
                torch.cuda.empty_cache()

        if len(results) == 2:
            speedup = results['Local']['tokens_per_sec'] / results['Full']['tokens_per_sec']
            mem_ratio = results['Local']['memory_mb'] / results['Full']['memory_mb']
            print(f"  Speedup: {speedup:.3f}x, Memory: {mem_ratio:.3f}x")


if __name__ == "__main__":
    benchmark_local_attention()