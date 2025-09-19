"""
GPT-2 MODEL WITH GATED ATTENTION
Based on "Gated Attention for Large Language Models" paper.

Key improvements:
- Eliminates attention sink (46.7% -> 4.8% on first token)
- Better training stability (allows 2x higher LR)
- 0.2+ PPL reduction
- 10+ point RULER improvement for context extension
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
class GatedConfig:
    # Model architecture
    vocab_size: int = 32768  # Mistral tokenizer size
    n_positions: int = 2048  # Max positions (can use shorter sequences)
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # For GQA

    # Gated attention parameters
    use_gated_attention: bool = True
    gate_type: str = "elementwise"  # "elementwise" or "headwise"
    gate_activation: str = "sigmoid"  # "sigmoid" or "silu"
    use_qk_norm: bool = False  # Optional QK normalization

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Layer configuration
    layernorm_epsilon: float = 1e-5
    use_rmsnorm: bool = True  # MUST use te.RMSNorm for FP8 compatibility
    norm_position: str = "pre"  # "pre" or "post"

    # MLP configuration
    ffn_hidden_size: Optional[int] = None
    mlp_type: str = "swiglu"

    # Position embeddings
    position_embedding_type: str = "learned"  # "learned" or "rope"
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Optimizations
    use_fused_qkv: bool = True
    use_fused_mlp: bool = False  # Benchmarks show it's slower
    use_pytorch_sdpa: bool = True
    use_fp8: bool = True

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = max(1, self.n_head // 4)  # Default GQA 4:1

        if self.ffn_hidden_size is None:
            if self.mlp_type in ["swiglu", "geglu"]:
                self.ffn_hidden_size = int(2 * 4 * self.n_embd / 3)
                self.ffn_hidden_size = (self.ffn_hidden_size + 255) // 256 * 256
            else:
                self.ffn_hidden_size = 4 * self.n_embd


class RotaryEmbedding(nn.Module):
    """RoPE implementation for position encoding."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x):
    """Helper for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply RoPE to Q and K."""
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GatedAttention(nn.Module):
    """
    Attention with gating mechanism based on the paper.
    Gate is applied after SDPA for maximum effectiveness.
    """

    def __init__(self, config: GatedConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.hidden_size // self.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.layer_idx = layer_idx

        # QK normalization (optional)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = te.RMSNorm(self.head_dim, eps=config.layernorm_epsilon)
            self.k_norm = te.RMSNorm(self.head_dim, eps=config.layernorm_epsilon)

        # Position embeddings
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.n_positions,
                base=config.rope_theta
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

        # Gating mechanism (the key innovation)
        self.use_gated_attention = config.use_gated_attention
        if self.use_gated_attention:
            self.gate_type = config.gate_type

            if self.gate_type == "elementwise":
                # Element-wise gating (best results in paper)
                gate_size = self.n_head * self.head_dim
            else:  # headwise
                # Head-wise gating (fewer parameters)
                gate_size = self.n_head

            self.gate_proj = te.Linear(self.hidden_size, gate_size, bias=False)

            if config.gate_activation == "sigmoid":
                self.gate_activation = nn.Sigmoid()
            else:  # silu
                self.gate_activation = nn.SiLU()

        self.out_proj = te.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.use_sdpa = config.use_pytorch_sdpa

    def forward(self, x, position_ids=None):
        S, B, _ = x.shape

        # Store normalized input for gating (query-dependent)
        gate_input = x

        # Fused QKV
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

        # GQA: repeat KV heads
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: [B, H, S, D]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Apply RoPE if configured
        if self.position_embedding_type == "rope":
            if position_ids is None:
                position_ids = torch.arange(0, S, dtype=torch.long, device=x.device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(v, seq_len=S)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # QK normalization (optional)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute attention
        if self.use_sdpa:
            # Simply use causal masking - SDPA will handle it efficiently
            # Padding is handled by the loss function (ignoring -100 labels)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # Let SDPA handle causal mask internally
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True  # Always causal for GPT
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)

        # Apply gating (the key innovation from the paper)
        if self.use_gated_attention:
            # Compute gate scores from normalized hidden states (query-dependent)
            gate_scores = self.gate_proj(gate_input)  # [S, B, gate_size]

            if self.gate_type == "elementwise":
                # Reshape for element-wise gating
                gate_scores = gate_scores.reshape(S, B, self.n_head, self.head_dim)
                gate_scores = gate_scores.permute(1, 2, 0, 3)  # [B, H, S, D]
            else:  # headwise
                # Reshape for head-wise gating
                gate_scores = gate_scores.reshape(S, B, self.n_head, 1)
                gate_scores = gate_scores.permute(1, 2, 0, 3)  # [B, H, S, 1]

            # Apply activation (sigmoid for sparsity)
            gate_scores = self.gate_activation(gate_scores)

            # Apply gate to attention output
            out = out * gate_scores

        # Reshape back
        out = out.permute(2, 0, 1, 3).reshape(S, B, self.hidden_size)
        out = self.out_proj(out)

        return out


class GatedMLP(nn.Module):
    """MLP layer (no fusion based on benchmarks)."""

    def __init__(self, config: GatedConfig):
        super().__init__()
        self.mlp_type = config.mlp_type

        if self.mlp_type == "vanilla":
            self.fc1 = te.Linear(config.n_embd, config.ffn_hidden_size, bias=True)
            self.fc2 = te.Linear(config.ffn_hidden_size, config.n_embd, bias=True)

        elif self.mlp_type in ["swiglu", "geglu"]:
            self.gate_proj = te.Linear(config.n_embd, config.ffn_hidden_size, bias=False)
            self.up_proj = te.Linear(config.n_embd, config.ffn_hidden_size, bias=False)
            self.down_proj = te.Linear(config.ffn_hidden_size, config.n_embd, bias=False)

    def forward(self, x):
        if self.mlp_type == "vanilla":
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)

        elif self.mlp_type == "swiglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up
            x = self.down_proj(x)

        elif self.mlp_type == "geglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.gelu(gate) * up
            x = self.down_proj(x)

        return x


class GatedTransformerBlock(nn.Module):
    """Transformer block with gated attention."""

    def __init__(self, config: GatedConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_position = config.norm_position

        # Normalization layers
        if config.use_rmsnorm:
            self.norm1 = te.RMSNorm(config.n_embd, eps=config.layernorm_epsilon)
            self.norm2 = te.RMSNorm(config.n_embd, eps=config.layernorm_epsilon)
        else:
            self.norm1 = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)
            self.norm2 = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)

        # Gated attention
        self.attn = GatedAttention(config, layer_idx)

        # MLP
        self.mlp = GatedMLP(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_ids=None):
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, position_ids)
        x = residual + self.dropout(x)

        # Pre-norm MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class GatedGPT2Model(nn.Module):
    """
    GPT-2 with Gated Attention.
    Implements the improvements from "Gated Attention for Large Language Models".
    """

    def __init__(self, config: GatedConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.position_embedding_type == "learned":
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GatedTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])

        # Final layer norm
        if config.use_rmsnorm:
            self.ln_f = te.RMSNorm(config.n_embd, eps=config.layernorm_epsilon)
        else:
            self.ln_f = te.LayerNorm(config.n_embd, eps=config.layernorm_epsilon)

        # Output projection (use te.Linear for FP8)
        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Convert to bfloat16 for better stability with FP8
        self.to(dtype=torch.bfloat16)

        # FP8 recipe (HYBRID for best gradient precision)
        self.fp8_recipe = DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID,
            amax_history_len=1024,
            amax_compute_algo="most_recent"
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, te.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        # Smaller init for large output layer (lm_head)
        if isinstance(module, te.Linear) and hasattr(module, 'out_features'):
            if module.out_features > 10000:
                nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, input_ids, position_ids=None, labels=None):
        B, S = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.wte(input_ids)

        # Position embeddings
        if self.config.position_embedding_type == "learned":
            if position_ids is None:
                position_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
            x = x + self.wpe(position_ids)

        x = self.drop(x)

        # Convert to [S, B, H] for TransformerEngine
        x = x.transpose(0, 1)

        # Apply transformer with FP8
        # Note: Padding is handled via labels=-100 in loss calculation
        with te.fp8_autocast(enabled=self.config.use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x, position_ids)

            x = self.ln_f(x)

            # Output projection should also be in FP8 context
            logits = self.lm_head(x)

        # Convert back to [B, S, V]
        logits = logits.transpose(0, 1)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.config.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100
            )

        return logits, loss


def get_gated_gpt2_config(
    vocab_size=32768,
    use_gated_attention=True,
    use_qk_norm=False,
    use_rope=True
):
    """Get configuration for gated GPT-2 model."""
    return GatedConfig(
        vocab_size=vocab_size,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=3,  # GQA 4:1

        # Gated attention settings
        use_gated_attention=use_gated_attention,
        gate_type="elementwise",  # Best from paper
        gate_activation="sigmoid",  # For sparsity
        use_qk_norm=use_qk_norm,

        # Position embeddings
        position_embedding_type="rope" if use_rope else "learned",
        rope_theta=10000.0,

        # Architecture
        mlp_type="swiglu",
        use_rmsnorm=True,
        norm_position="pre",

        # Optimizations
        use_fused_qkv=True,
        use_fused_mlp=False,
        use_pytorch_sdpa=True,
        use_fp8=True
    )