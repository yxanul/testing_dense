"""
GPT-2 with FlashAttention 2 support using TransformerEngine.
Automatically uses the best available attention backend.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from dataclasses import dataclass
from typing import Optional

# Enable FlashAttention if available
os.environ["NVTE_FLASH_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN"] = "0"  # Prefer Flash over cuDNN


@dataclass
class FlashConfig:
    vocab_size: int = 50304
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # For GQA
    dropout: float = 0.1
    attn_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5

    # Attention backend
    use_flash_attn: bool = True  # Try to use FlashAttention
    use_te_attention: bool = True  # Use TE's optimized attention

    # FP8 settings
    use_fp8: bool = True
    fp8_margin: int = 0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head


class FlashTransformerBlock(nn.Module):
    """Transformer block with FlashAttention support."""

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.hidden_size // self.n_head

        # Fused LayerNorm + QKV
        self.ln_qkv = te.LayerNormLinear(
            self.hidden_size,
            3 * self.hidden_size if self.n_kv_head == self.n_head
            else self.hidden_size + 2 * self.n_kv_head * self.head_dim,
            eps=config.layernorm_epsilon,
            bias=True
        )

        # Use TE's optimized attention modules
        if config.use_te_attention:
            # Option 1: MultiheadAttention (auto-selects best backend)
            self.attention = te.attention.MultiheadAttention(
                hidden_size=self.hidden_size,
                num_attention_heads=self.n_head,
                kv_channels=self.head_dim,
                attention_dropout=config.attn_dropout,
                # Set backend preference
                attn_mask_type="causal",  # Built-in causal mask
            )
            self.use_mha = True
        else:
            # Option 2: DotProductAttention (more control)
            self.attention = te.attention.DotProductAttention(
                num_attention_heads=self.n_head,
                kv_channels=self.head_dim,
                attention_dropout=config.attn_dropout,
                attn_mask_type="causal",
            )
            self.use_mha = False

        # Output projection
        self.proj = te.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        # Fused LayerNorm + MLP
        self.ln_mlp = te.LayerNormMLP(
            self.hidden_size,
            4 * self.hidden_size,
            eps=config.layernorm_epsilon,
            bias=True
        )

    def forward(self, x):
        # x: [seq, batch, hidden]
        S, B, _ = x.shape
        residual = x

        # Fused LayerNorm + QKV projection
        qkv = self.ln_qkv(x)  # [S, B, 3*H or H+2*KV_H]

        if self.use_mha:
            # MultiheadAttention handles everything internally
            attn_out = self.attention(qkv, attention_mask=None)
        else:
            # Reshape for DotProductAttention
            if self.n_kv_head == self.n_head:
                # Standard MHA
                qkv = qkv.view(S, B, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            else:
                # GQA: separate Q from KV
                q_size = self.n_head * self.head_dim
                kv_size = self.n_kv_head * self.head_dim

                q = qkv[:, :, :q_size].view(S, B, self.n_head, self.head_dim)
                k = qkv[:, :, q_size:q_size + kv_size].view(S, B, self.n_kv_head, self.head_dim)
                v = qkv[:, :, q_size + kv_size:].view(S, B, self.n_kv_head, self.head_dim)

                # Repeat KV for GQA
                if self.n_kv_head < self.n_head:
                    k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
                    v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)

            # Run attention
            attn_out = self.attention(q, k, v)

        # Projection + residual
        attn_out = self.proj(attn_out)
        x = residual + self.dropout(attn_out)

        # MLP + residual
        residual = x
        x = self.ln_mlp(x)
        x = residual + x

        return x


class ManualFlashAttention(nn.Module):
    """Manual FlashAttention implementation as fallback."""

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.attn_dropout

        # Try to import flash_attn
        self.use_flash = False
        try:
            from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
            self.use_flash = True
            print("✓ Using FlashAttention 2")
        except ImportError:
            print("✗ FlashAttention not available, using standard attention")

    def forward(self, q, k, v):
        """
        q, k, v: [batch, seq, heads, dim] or [seq, batch, heads, dim]
        """
        if self.use_flash:
            # FlashAttention expects [batch, seq, heads, dim]
            if q.dim() == 4 and q.shape[0] != q.shape[1]:
                # Convert from [seq, batch, heads, dim] to [batch, seq, heads, dim]
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
                need_transpose = True
            else:
                need_transpose = False

            # Run FlashAttention
            out = self.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )

            if need_transpose:
                out = out.transpose(0, 1)

            return out
        else:
            # Fallback to standard attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            S = q.shape[-3] if q.dim() == 4 else q.shape[-2]
            causal_mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)

            out = torch.matmul(attn, v)
            return out


class GPT2FlashModel(nn.Module):
    """GPT-2 with FlashAttention support."""

    def __init__(self, config: FlashConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FlashTransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)
        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)
        self.to(dtype=torch.bfloat16)

        # FP8 recipe
        self.fp8_recipe = DelayedScaling(
            margin=config.fp8_margin,
            fp8_format=Format.HYBRID  # E4M3 fwd, E5M2 bwd
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, 0, 0.02)
        elif isinstance(module, te.Linear):
            nn.init.normal_(module.weight, 0, 0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.drop(x)
        x = x.transpose(0, 1)  # [S, B, H]

        # Transformer blocks with FP8
        with te.fp8_autocast(enabled=self.config.use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

        return logits.transpose(0, 1)  # [B, S, V]


if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    print("Testing GPT-2 with FlashAttention:")
    print("=" * 60)

    # Check if FlashAttention is available
    try:
        import flash_attn
        print(f"✓ FlashAttention {flash_attn.__version__} available")
    except ImportError:
        print("✗ FlashAttention not installed")
        print("  Install with: pip install flash-attn --no-build-isolation")

    # Test configurations
    configs = [
        ("TE MultiheadAttention", FlashConfig(n_layer=2, use_te_attention=True)),
        ("TE with GQA", FlashConfig(n_layer=2, n_kv_head=4, use_te_attention=True)),
    ]

    B, S = 8, 512
    vocab_size = 50304

    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 40)

        model = GPT2FlashModel(config).to(device)

        x = torch.randint(0, vocab_size, (B, S), device=device)
        y = torch.randint(0, vocab_size, (B, S), device=device)

        # Warmup
        for _ in range(3):
            logits = model(x)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 100  # ms per iteration

        print(f"✓ Time per iteration: {elapsed:.2f} ms")
        print(f"  Loss: {loss.item():.4f}")

        # Memory usage
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {mem_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()