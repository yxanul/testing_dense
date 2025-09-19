"""
GPT-2 sized decoder-only Transformer built on NVIDIA TransformerEngine with FP8.

Tested against TransformerEngine 2.x APIs.

Quick start (CUDA 12.2+ on Hopper/Ada/Blackwell GPU, PyTorch ≥2.3 recommended):

    pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
    pip install --no-build-isolation "transformer_engine[pytorch]"

Optional environment toggles:
  * NVTE_FLASH_ATTN=1 to prefer FlashAttention backend if available
  * NVTE_FUSED_ATTN=1 to use cuDNN fused attention backend

Run a quick synthetic forward/backward:

    python gpt2_te_fp8.py

Notes
-----
- We wrap the model's forward pass in `te.fp8_autocast(...)` so attention/MLP
  math uses FP8 with runtime-managed scaling; weights stay in BF16 by default.
- Shape convention inside the model matches TE tutorials: [seq, batch, hidden].
- This is a minimal, HF-independent GPT-2 small (124M-ish) config by default.
- Generation here recomputes full context each step (no KV cache) to keep the
  example simple; swap in TE InferenceParams if you want fast decoding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format


# -----------------------------
# Config
# -----------------------------
@dataclass
class GPT2Config:
    vocab_size: int = 50264
    n_positions: int = 1024
    n_embd: int = 768          # GPT-2 small
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1       # embedding/residual dropout
    attn_dropout: float = 0.0  # attention dropout
    layernorm_epsilon: float = 1e-5
    tie_embeddings: bool = False

    # TransformerEngine / precision knobs
    weights_dtype: torch.dtype = torch.bfloat16
    use_fp8: bool = True
    # Force classic FP8 (no MXFP8) by default. You can still override via fp8_recipe.
    # HYBRID = E4M3 for fwd, E5M2 for bwd/grad (works well on H100/Blackwell).
    fp8_format: Format = Format.HYBRID
    fp8_recipe: Optional[DelayedScaling] = None  # if None, we'll build one from fp8_format  # None => use default recipe


# -----------------------------
# Model
# -----------------------------
class GPT2TEModel(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg

        # Token & position embeddings (learned, like GPT-2)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # Stack of fused TransformerEngine layers. Defaults to causal masking.
        self.blocks = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=cfg.n_embd,
                ffn_hidden_size=4 * cfg.n_embd,
                num_attention_heads=cfg.n_head,
                layernorm_epsilon=cfg.layernorm_epsilon,
                attention_dropout=cfg.attn_dropout,
                hidden_dropout=cfg.dropout,
                # Default self_attn_mask_type is 'causal'; leave as default.
            )
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = te.LayerNorm(cfg.n_embd, eps=cfg.layernorm_epsilon)
        self.lm_head = te.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Init + dtype
        self.apply(self._init_weights)
        self.to(dtype=cfg.weights_dtype)

        # (Optional) tie output projection to embeddings. Works if TE Linear
        # exposes .weight as Parameter shaped [vocab, hidden]. If your TE
        # version doesn't support tying, set tie_embeddings=False.
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight  # type: ignore[attr-defined]

        # FP8 recipe
        # FP8 recipe: use classic FP8 (DelayedScaling) explicitly, NOT MXFP8.
        self._fp8_recipe = cfg.fp8_recipe or DelayedScaling(fp8_format=cfg.fp8_format)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Embedding,)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # TE Linear exposes weight/bias like nn.Linear
        linear_types = (te.Linear,)
        if isinstance(module, linear_types):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,           # [batch, seq]
        attention_mask: Optional[torch.Tensor] = None,  # padding mask (optional)
        use_fp8: Optional[bool] = None,
    ) -> torch.Tensor:                         # logits [batch, seq, vocab]
        """Forward pass with FP8 autocast for TE modules.

        `attention_mask` is ignored for causal-only masking (default). If you
        pass a padding mask, set it to shape [batch, 1, 1, seq] with True for
        masked positions.
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        # Embeddings (batch-first), then switch to [seq, batch, hidden]
        pos_ids = torch.arange(0, seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.emb_drop(x)
        x = x.transpose(0, 1).contiguous()  # [S, B, H]

        # Enable FP8 for TE modules during forward
        fp8_on = self.cfg.use_fp8 if use_fp8 is None else use_fp8
        with te.fp8_autocast(enabled=fp8_on, fp8_recipe=self._fp8_recipe):
            for block in self.blocks:
                # Causal masking is internal; we only pass attention_mask when
                # it encodes padding/arbitrary masks.
                x = block(x, attention_mask=attention_mask)
            x = self.ln_f(x)
            logits = self.lm_head(x)  # [S, B, V]

        return logits.transpose(0, 1).contiguous()  # [B, S, V]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """Greedy/top-k sampling without KV cache for simplicity.
        Recomputes full context each step.
        """
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out)[:, -1, :]  # [B, V]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_keep = v[..., -1, None]
                logits = torch.where(logits < min_keep, torch.full_like(logits, float('-inf')), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            out = torch.cat([out, next_token], dim=1)
        return out


# -----------------------------
# Tiny test / demo
# -----------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPT2Config()
    model = GPT2TEModel(cfg).to(device)

    # Synthetic batch
    B, S = 2, 128
    x = torch.randint(0, cfg.vocab_size, (B, S), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    # Forward in FP8 (default)
    logits = model(x)  # [B, S, V]

    # Simple language-modeling loss
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

    # Backprop
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # One short generate step
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    out_tokens = model.generate(prompt, max_new_tokens=8)
    print({"loss": float(loss.detach().cpu()), "prompt": prompt.tolist(), "out": out_tokens.tolist()})
