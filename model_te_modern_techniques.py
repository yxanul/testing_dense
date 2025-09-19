"""
Modern LLM Techniques Testing Suite
Implements techniques from LLaMA, Qwen, Phi, Mistral, etc.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class ModernConfig:
    # Model architecture
    vocab_size: int = 32768
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None

    # Modern techniques
    position_embedding_type: str = "rope"  # "learned", "rope", "alibi", "none"
    rope_theta: float = 10000.0  # LLaMA uses 10000, CodeLLaMA uses 1000000
    rope_scaling: Optional[dict] = None  # For extended context (LLaMA 2 Long)

    mlp_type: str = "swiglu"  # "vanilla", "swiglu", "geglu", "reglu"
    mlp_bias: bool = False  # Most modern models don't use bias

    attention_type: str = "gqa"  # "mha", "mqa", "gqa"
    attention_bias: bool = False  # Phi uses True, LLaMA uses False

    norm_type: str = "rmsnorm"  # "layernorm", "rmsnorm"
    norm_position: str = "pre"  # "pre", "post", "sandwich"
    norm_eps: float = 1e-5  # LLaMA uses 1e-5, some use 1e-6

    # Phi-specific
    partial_rotary_factor: float = 1.0  # Phi uses 0.4 (only rotate 40% of head dim)

    # Mistral-specific
    sliding_window: Optional[int] = None  # Mistral uses 4096

    # Qwen-specific
    use_qk_norm: bool = False  # Normalize Q and K for stability

    # Training stability
    use_parallel_residual: bool = False  # GPT-J/Pythia style
    hidden_dropout: float = 0.0  # Most modern models use 0
    attention_dropout: float = 0.0

    # Optimizations
    use_flash_attention: bool = True
    use_fp8: bool = True
    use_fused_qkv: bool = True

    def __post_init__(self):
        if self.n_kv_head is None:
            if self.attention_type == "mha":
                self.n_kv_head = self.n_head
            elif self.attention_type == "mqa":
                self.n_kv_head = 1  # Single KV head
            else:  # gqa
                self.n_kv_head = self.n_head // 4  # 4:1 ratio

        # Calculate MLP hidden size
        if self.mlp_type in ["swiglu", "geglu", "reglu"]:
            # GLU variants use 2/3 ratio
            intermediate_size = int(2 * 4 * self.n_embd / 3)
            self.ffn_hidden_size = (intermediate_size + 255) // 256 * 256
        else:
            self.ffn_hidden_size = 4 * self.n_embd


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) from LLaMA."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for maximum sequence length
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[0]

        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """Helper for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply RoPE to queries and keys."""
    # Handle different dimensions for partial rotary
    if q.shape[-1] != cos.shape[-1]:
        # This shouldn't happen with the fix, but add safety check
        cos = cos[..., :q.shape[-1]]
        sin = sin[..., :q.shape[-1]]

    cos = cos.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1, dim]
    sin = sin.unsqueeze(1).unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernAttention(nn.Module):
    """Modern attention with RoPE, QK-Norm, MQA/GQA support."""

    def __init__(self, config: ModernConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.hidden_size // self.n_head
        self.layer_idx = layer_idx

        # Attention type
        self.attention_type = config.attention_type
        self.n_rep = self.n_head // self.n_kv_head if self.n_kv_head else 1

        # Position embeddings
        self.position_embedding_type = config.position_embedding_type
        self.partial_rotary_factor = config.partial_rotary_factor

        if self.position_embedding_type == "rope":
            # For partial rotary, we only rotate part of the head dimensions
            self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
            self.rotary_emb = RotaryEmbedding(
                self.rope_dim,
                max_position_embeddings=config.n_positions,
                base=config.rope_theta,
            )

        # QK Normalization (Qwen2)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = te.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = te.RMSNorm(self.head_dim, eps=config.norm_eps)

        # Projections
        if config.use_fused_qkv and self.attention_type == "mha":
            # Fused QKV for MHA
            self.qkv_proj = te.Linear(
                self.hidden_size,
                3 * self.hidden_size,
                bias=config.attention_bias
            )
        else:
            # Separate projections for MQA/GQA
            self.q_proj = te.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
            self.k_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=config.attention_bias)
            self.v_proj = te.Linear(self.hidden_size, self.n_kv_head * self.head_dim, bias=config.attention_bias)

        self.out_proj = te.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.sliding_window = config.sliding_window

    def forward(self, x, attention_mask=None):
        S, B, _ = x.shape

        # Get Q, K, V
        if hasattr(self, 'qkv_proj'):
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Reshape
        q = q.reshape(S, B, self.n_head, self.head_dim)
        k = k.reshape(S, B, self.n_kv_head, self.head_dim)
        v = v.reshape(S, B, self.n_kv_head, self.head_dim)

        # Apply RoPE if configured
        if self.position_embedding_type == "rope":
            if self.partial_rotary_factor < 1.0:
                # Partial rotary: only rotate part of the dimensions
                q_rot = q[..., :self.rope_dim]
                q_pass = q[..., self.rope_dim:]
                k_rot = k[..., :self.rope_dim]
                k_pass = k[..., self.rope_dim:]

                cos, sin = self.rotary_emb(q_rot, seq_len=S)
                q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

                # Concatenate rotated and non-rotated parts
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)
            else:
                # Full rotary
                cos, sin = self.rotary_emb(q, seq_len=S)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # QK Normalization (for training stability)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Handle MQA/GQA
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: [B, H, S, D]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Compute attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=True if attention_mask is None else False,
        )

        # Reshape output
        out = out.permute(2, 0, 1, 3).reshape(S, B, self.hidden_size)
        out = self.out_proj(out)

        return out


class ModernMLP(nn.Module):
    """Modern MLP with various activation functions."""

    def __init__(self, config: ModernConfig):
        super().__init__()
        self.mlp_type = config.mlp_type

        if self.mlp_type == "vanilla":
            self.fc1 = te.Linear(config.n_embd, config.ffn_hidden_size, bias=config.mlp_bias)
            self.fc2 = te.Linear(config.ffn_hidden_size, config.n_embd, bias=config.mlp_bias)
        else:
            # GLU variants
            self.gate_proj = te.Linear(config.n_embd, config.ffn_hidden_size, bias=config.mlp_bias)
            self.up_proj = te.Linear(config.n_embd, config.ffn_hidden_size, bias=config.mlp_bias)
            self.down_proj = te.Linear(config.ffn_hidden_size, config.n_embd, bias=config.mlp_bias)

    def forward(self, x):
        if self.mlp_type == "vanilla":
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
        elif self.mlp_type == "swiglu":
            gate = F.silu(self.gate_proj(x))
            x = gate * self.up_proj(x)
            x = self.down_proj(x)
        elif self.mlp_type == "geglu":
            gate = F.gelu(self.gate_proj(x))
            x = gate * self.up_proj(x)
            x = self.down_proj(x)
        elif self.mlp_type == "reglu":
            gate = F.relu(self.gate_proj(x))
            x = gate * self.up_proj(x)
            x = self.down_proj(x)

        return x


class ModernTransformerBlock(nn.Module):
    """Modern transformer block with various norm positions."""

    def __init__(self, config: ModernConfig, layer_idx: int = 0):
        super().__init__()

        # Choose normalization
        if config.norm_type == "rmsnorm":
            self.norm1 = te.RMSNorm(config.n_embd, eps=config.norm_eps)
            self.norm2 = te.RMSNorm(config.n_embd, eps=config.norm_eps)
            if config.norm_position == "sandwich":
                self.norm3 = te.RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.norm1 = te.LayerNorm(config.n_embd, eps=config.norm_eps)
            self.norm2 = te.LayerNorm(config.n_embd, eps=config.norm_eps)
            if config.norm_position == "sandwich":
                self.norm3 = te.LayerNorm(config.n_embd, eps=config.norm_eps)

        self.attn = ModernAttention(config, layer_idx)
        self.mlp = ModernMLP(config)

        self.norm_position = config.norm_position
        self.use_parallel_residual = config.use_parallel_residual
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x, attention_mask=None):
        if self.norm_position == "pre":
            # Pre-norm (LLaMA style)
            residual = x
            x = self.norm1(x)
            x = self.attn(x, attention_mask)
            x = residual + self.hidden_dropout(x)

            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + self.hidden_dropout(x)

        elif self.norm_position == "post":
            # Post-norm (original transformer)
            residual = x
            x = self.attn(x, attention_mask)
            x = residual + self.hidden_dropout(x)
            x = self.norm1(x)

            residual = x
            x = self.mlp(x)
            x = residual + self.hidden_dropout(x)
            x = self.norm2(x)

        elif self.norm_position == "sandwich":
            # Sandwich norm (some experimental models)
            x = self.norm1(x)
            residual = x
            x = self.attn(x, attention_mask)
            x = self.norm3(x)
            x = residual + self.hidden_dropout(x)

            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + self.hidden_dropout(x)

        return x


class ModernGPT2(nn.Module):
    """GPT-2 with modern techniques from various LLMs."""

    def __init__(self, config: ModernConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings (if using learned)
        if config.position_embedding_type == "learned":
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.hidden_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])

        # Final norm
        if config.norm_type == "rmsnorm":
            self.ln_f = te.RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_f = te.LayerNorm(config.n_embd, eps=config.norm_eps)

        # Output projection
        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # Initialize
        self.apply(self._init_weights)

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

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.wte(input_ids)

        # Position embeddings
        if self.config.position_embedding_type == "learned":
            pos_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
            x = x + self.wpe(pos_ids)

        x = self.drop(x)

        # Convert to [S, B, H] for TE modules
        x = x.transpose(0, 1)

        # Apply transformer blocks
        with te.fp8_autocast(enabled=self.config.use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x, attention_mask)

            x = self.ln_f(x)
            logits = self.lm_head(x)

        # Convert back to [B, S, V]
        logits = logits.transpose(0, 1)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1)
            )

        return logits, loss


# Configuration presets for different models
def get_llama_config():
    """LLaMA-style configuration."""
    return ModernConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,  # LLaMA 1 uses MHA, LLaMA 2 uses GQA
        position_embedding_type="rope",
        rope_theta=10000.0,
        mlp_type="swiglu",
        mlp_bias=False,
        attention_bias=False,
        norm_type="rmsnorm",
        norm_position="pre",
        norm_eps=1e-5,
    )


def get_mistral_config():
    """Mistral-style configuration."""
    return ModernConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=3,  # GQA 4:1
        position_embedding_type="rope",
        rope_theta=10000.0,
        sliding_window=512,  # Mistral uses 4096 but we'll use 512 for testing
        mlp_type="swiglu",
        mlp_bias=False,
        attention_bias=False,
        norm_type="rmsnorm",
        norm_position="pre",
    )


def get_phi_config():
    """Phi-style configuration."""
    return ModernConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,  # MHA
        position_embedding_type="rope",
        partial_rotary_factor=0.4,  # Only rotate 40% of dims
        mlp_type="geglu",  # Phi uses GELU-based GLU
        mlp_bias=True,  # Phi uses bias
        attention_bias=True,
        norm_type="layernorm",
        norm_position="pre",
    )


def get_qwen_config():
    """Qwen-style configuration."""
    return ModernConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,  # Qwen2 uses GQA in larger models
        position_embedding_type="rope",
        rope_theta=1000000.0,  # Qwen uses larger theta
        use_qk_norm=True,  # QK normalization for stability
        mlp_type="swiglu",
        mlp_bias=False,
        attention_bias=True,
        norm_type="rmsnorm",
        norm_position="pre",
    )


def get_falcon_config():
    """Falcon-style configuration (MQA)."""
    return ModernConfig(
        n_layer=12,
        n_embd=768,
        n_head=12,
        attention_type="mqa",  # Multi-Query Attention
        n_kv_head=1,  # Single KV head for all queries
        position_embedding_type="rope",
        mlp_type="gelu",  # Falcon uses standard GELU
        norm_type="layernorm",
        norm_position="pre",
        use_parallel_residual=True,  # Parallel attention/MLP
    )


def benchmark_modern_techniques():
    """Benchmark different modern LLM techniques."""
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("MODERN LLM TECHNIQUES BENCHMARK")
    print("=" * 80)

    configs = [
        ("LLaMA-style", get_llama_config()),
        ("Mistral-style", get_mistral_config()),
        ("Phi-style", get_phi_config()),
        ("Qwen-style", get_qwen_config()),
        ("Falcon-style (MQA)", get_falcon_config()),
    ]

    batch_size = 8
    seq_len = 1024

    results = {}

    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 60)
        print(f"  Position: {config.position_embedding_type}")
        print(f"  Attention: {config.attention_type} (KV heads: {config.n_kv_head})")
        print(f"  MLP: {config.mlp_type}")
        print(f"  Norm: {config.norm_type} ({config.norm_position})")

        if config.use_qk_norm:
            print(f"  QK Norm: Enabled")
        if config.sliding_window:
            print(f"  Sliding Window: {config.sliding_window}")
        if config.partial_rotary_factor < 1.0:
            print(f"  Partial RoPE: {config.partial_rotary_factor}")

        try:
            model = ModernGPT2(config).to(device).to(torch.bfloat16)

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params/1e6:.1f}M")

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

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

            results[name] = {
                'tokens_per_sec': tokens_per_sec,
                'memory_mb': memory_mb,
                'ms_per_iter': avg_time * 1000,
                'params_m': n_params / 1e6
            }

            print(f"  Speed: {tokens_per_sec:,.0f} tokens/sec")
            print(f"  Memory: {memory_mb:.0f} MB")
            print(f"  ms/iter: {avg_time * 1000:.1f} ms")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        # Find best
        best_speed = max(results.items(), key=lambda x: x[1]['tokens_per_sec'])
        best_memory = min(results.items(), key=lambda x: x[1]['memory_mb'])

        print(f"\nFastest: {best_speed[0]} ({best_speed[1]['tokens_per_sec']:,.0f} tok/s)")
        print(f"Most memory efficient: {best_memory[0]} ({best_memory[1]['memory_mb']:.0f} MB)")

        # Ranking
        print("\nSpeed Ranking:")
        for i, (name, res) in enumerate(sorted(results.items(), key=lambda x: x[1]['tokens_per_sec'], reverse=True), 1):
            print(f"  {i}. {name:<20} {res['tokens_per_sec']:>8,.0f} tok/s")

        print("\nKey Findings:")

        # MQA vs GQA vs MHA
        mqa_result = results.get("Falcon-style (MQA)")
        base_result = results.get("LLaMA-style")
        if mqa_result and base_result:
            speedup = mqa_result['tokens_per_sec'] / base_result['tokens_per_sec']
            print(f"  MQA vs MHA speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_modern_techniques()