"""
Advanced GPT-2 with customizable features using TransformerEngine fused modules.
Supports: RMSNorm, QK normalization, Grouped Query Attention (GQA), and more.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdvancedConfig:
    vocab_size: int = 50304  # Must be divisible by 32 for FP8
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # For GQA/MQA, None means equal to n_head
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-5

    # Advanced features
    use_rmsnorm: bool = False  # Use RMSNorm instead of LayerNorm
    use_qk_norm: bool = False  # Apply normalization to Q and K
    qk_norm_scale: float = 768 ** -0.5  # Scale factor for QK norm
    use_rotary: bool = False  # Use RoPE positional encoding
    rope_theta: float = 10000.0

    # FP8 settings
    use_fp8: bool = True
    fp8_margin: int = 0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"


class RMSNorm(nn.Module):
    """RMSNorm implementation compatible with TE modules."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        # x: [batch, heads, seq, head_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class AdvancedTransformerBlock(nn.Module):
    """Advanced transformer block with GQA, RMSNorm, QK norm, etc."""

    def __init__(self, config: AdvancedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.hidden_size // self.n_head
        self.n_rep = self.n_head // self.n_kv_head  # Repetition factor for GQA

        # Choose normalization type
        if config.use_rmsnorm:
            self.norm1 = RMSNorm(self.hidden_size, config.layernorm_epsilon)
            self.norm2 = RMSNorm(self.hidden_size, config.layernorm_epsilon)
        else:
            self.norm1 = te.LayerNorm(self.hidden_size, eps=config.layernorm_epsilon)
            self.norm2 = te.LayerNorm(self.hidden_size, eps=config.layernorm_epsilon)

        # QKV projection with GQA support
        self.q_proj = te.Linear(self.hidden_size, self.n_head * self.head_dim, bias=True)
        self.k_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=True)
        self.v_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=True)

        # Optional QK normalization
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.layernorm_epsilon)
            self.k_norm = RMSNorm(self.head_dim, config.layernorm_epsilon)

        # Rotary embeddings
        if config.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                config.n_positions,
                config.rope_theta
            )

        # Output projection
        self.proj = te.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

        # MLP with fused operations
        if config.use_rmsnorm:
            # Can't use LayerNormMLP with RMSNorm, use separate components
            self.mlp = nn.Sequential(
                te.Linear(self.hidden_size, 4 * self.hidden_size, bias=True),
                nn.GELU(),
                te.Linear(4 * self.hidden_size, self.hidden_size, bias=True)
            )
        else:
            self.mlp = te.LayerNormMLP(
                self.hidden_size, 4 * self.hidden_size,
                eps=config.layernorm_epsilon, bias=True
            )

    def forward(self, x):
        # x: [seq, batch, hidden]
        S, B, _ = x.shape
        residual = x

        # Pre-norm
        x = self.norm1(x)

        # Compute Q, K, V
        q = self.q_proj(x)  # [S, B, n_head * head_dim]
        k = self.k_proj(x)  # [S, B, n_kv_head * head_dim]
        v = self.v_proj(x)  # [S, B, n_kv_head * head_dim]

        # Reshape for attention
        q = q.view(S, B, self.n_head, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(S, B, self.n_kv_head, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(S, B, self.n_kv_head, self.head_dim).permute(1, 2, 0, 3)

        # GQA: Repeat K and V if needed
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Apply QK normalization if enabled
        if self.config.use_qk_norm:
            q_shape = q.shape
            k_shape = k.shape
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary:
            cos, sin = self.rotary_emb(q, S)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.config.use_qk_norm:
            scale = self.config.qk_norm_scale

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_out = torch.matmul(attn_weights, v)  # [B, heads, S, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.hidden_size)
        attn_out = attn_out.transpose(0, 1)  # [S, B, H]

        # Output projection + residual
        attn_out = self.proj(attn_out)
        x = residual + self.dropout(attn_out)

        # MLP block
        residual = x
        if self.config.use_rmsnorm:
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + x
        else:
            x = self.mlp(x)  # LayerNormMLP includes the residual
            x = residual + x

        return x


class AdvancedGPT2Model(nn.Module):
    """Advanced GPT-2 with all customizable features."""

    def __init__(self, config: AdvancedConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if not config.use_rotary:
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        # Final norm
        if config.use_rmsnorm:
            self.ln_f = RMSNorm(config.n_embd, config.layernorm_epsilon)
        else:
            self.ln_f = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)

        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)
        self.to(dtype=torch.bfloat16)

        # FP8 recipe
        self.fp8_recipe = DelayedScaling(
            margin=config.fp8_margin,
            fp8_format=Format.E4M3
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, te.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        # Small init for large output layer
        if isinstance(module, te.Linear) and hasattr(module, 'out_features'):
            if module.out_features > 10000:
                nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        x = self.wte(input_ids)
        if not self.config.use_rotary:
            pos_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
            x = x + self.wpe(pos_ids)
        x = self.drop(x)
        x = x.transpose(0, 1)  # [S, B, H]

        # Transformer blocks with FP8
        with te.fp8_autocast(enabled=self.config.use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)  # [S, B, V]

        return logits.transpose(0, 1)  # [B, S, V]


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    print("Testing Advanced GPT-2 configurations:")
    print("=" * 60)

    configs = [
        ("Standard GPT-2", AdvancedConfig(n_layer=2)),
        ("With RMSNorm", AdvancedConfig(n_layer=2, use_rmsnorm=True)),
        ("With GQA (4 KV heads)", AdvancedConfig(n_layer=2, n_head=12, n_kv_head=4)),
        ("With QK Norm", AdvancedConfig(n_layer=2, use_qk_norm=True)),
        ("With RoPE", AdvancedConfig(n_layer=2, use_rotary=True)),
        ("All features", AdvancedConfig(
            n_layer=2,
            use_rmsnorm=True,
            n_kv_head=4,
            use_qk_norm=True,
            use_rotary=True
        )),
    ]

    B, S = 2, 128
    vocab_size = 50304

    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 40)

        model = AdvancedGPT2Model(config).to(device)

        x = torch.randint(0, vocab_size, (B, S), device=device)
        y = torch.randint(0, vocab_size, (B, S), device=device)

        try:
            # Forward with FP8
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            # Backward
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"✓ Loss: {loss.item():.4f}")

            # Model stats
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params/1e6:.1f}M")
            print(f"  GQA ratio: {config.n_head}/{config.n_kv_head}")

        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")