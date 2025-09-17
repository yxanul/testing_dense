#!/usr/bin/env python3
"""
Minimal BF16 Transformer with MoE (top-1) for fast experiments.

Targets:
- 10–12 layers, d_model 512–1024
- SwiGLU FFN (≈2.67x expansion, i.e., 8/3)
- MoE with top-1 routing, optional dropless routing
- Load-balancing auxiliary loss (0.01–0.1)
- Capacity factor 1.0–1.25 (used only if dropless=False)
- AdamW, RMSNorm, BF16 forward
- Vocab size 32768 by default (matches your setup)

Notes on expert size (SwiGLU MoE): params_per_expert ≈ 8*d^2.
- d=512  -> ~2.1M
- d=768  -> ~4.7M
- d=1024 -> ~8.4M

Default config here aims <~125M params for quick iteration:
- d=512, n_layer=10, n_head=8, n_experts=4 (MoE in every block)
- Rough total ≈ 110–115M with tied embeddings.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from transformers import AutoTokenizer


import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch._dynamo as _dynamo
    _dynamo_disable = _dynamo.disable
except Exception:
    def _dynamo_disable(fn):
        return fn

# ----------------------- Optional FP8 helpers -----------------------

_FP8_AVAILABLE = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, '_scaled_mm')

if _FP8_AVAILABLE:
    from torch import Tensor

    try:
        import torch.library as _lib
    except Exception:
        _lib = None  # pragma: no cover

    if _lib is not None:
        @_lib.custom_op("nanogpt::mm", mutates_args=())  # type: ignore
        def _fp8_mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> Tuple[Tensor, Tensor, Tensor]:
            @torch.compile
            def impl(x: Tensor, w: Tensor):
                assert x.is_contiguous() and w.is_contiguous()
                x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
                w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
                out = torch._scaled_mm(
                    x_f8,
                    w_f8.T,
                    out_dtype=torch.bfloat16,
                    scale_a=x.new_tensor(x_s, dtype=torch.float32),
                    scale_b=x.new_tensor(w_s, dtype=torch.float32),
                    use_fast_accum=True,
                )
                return out, x_f8, w_f8

            return impl(x, w)

        @_fp8_mm_op.register_fake  # type: ignore
        def _(x: Tensor, w: Tensor, *_):
            assert x.ndim == 2 and w.ndim == 2 and x.shape[1] == w.shape[1]
            assert x.device == w.device
            assert x.is_contiguous() and w.is_contiguous()
            return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

        @_lib.custom_op("nanogpt::mm_backward", mutates_args=())  # type: ignore
        def _fp8_mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> Tuple[Tensor, Tensor]:
            @torch.compile
            def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
                assert grad.is_contiguous()
                x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
                w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
                grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
                grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
                grad_x = torch._scaled_mm(
                    grad_f8,
                    w_f8.T.contiguous().T,
                    out_dtype=torch.bfloat16,
                    scale_a=grad_inv_s,
                    scale_b=w_inv_s,
                    use_fast_accum=False,
                )
                grad_w = torch._scaled_mm(
                    x_f8.T.contiguous(),
                    grad_f8.T.contiguous().T,
                    out_dtype=torch.float32,
                    scale_a=x_inv_s,
                    scale_b=grad_inv_s,
                    use_fast_accum=False,
                ).T
                return grad_x, grad_w

            return impl(g, x_f8, w_f8)

        @_fp8_mm_backward_op.register_fake  # type: ignore
        def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
            return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

        def _fp8_mm_backward(ctx, grad_out: Tensor, *_):
            x_f8, w_f8 = ctx.saved_tensors
            x_s, w_s, grad_s = ctx.scales
            grad_x, grad_w = torch.ops.nanogpt.mm_backward(  # type: ignore
                grad_out.contiguous(), x_f8, w_f8, x_s, w_s, grad_s
            )
            return grad_x, grad_w, None, None, None

        def _fp8_mm_setup(ctx: torch.autograd.function.FunctionCtx, inputs, output):
            *_, x_s, w_s, grad_s = inputs
            _, x_f8, w_f8 = output
            ctx.save_for_backward(x_f8, w_f8)
            ctx.scales = x_s, w_s, grad_s
            ctx.set_materialize_grads(False)

        _fp8_mm_op.register_autograd(_fp8_mm_backward, setup_context=_fp8_mm_setup)  # type: ignore


class FP8Linear(nn.Module):
    """Minimal FP8 Linear: FP8 matmul with BF16 master weights/accum.

    Falls back to nn.Linear if FP8 is not available.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16)) if bias else None
        # simple per-tensor scales
        self.register_buffer('x_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('w_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('g_scale', torch.tensor(1.0, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _FP8_AVAILABLE or not x.is_cuda:
            y = torch.matmul(x, self.weight.T)
            if self.bias is not None:
                y = y + self.bias
            return y
        # update scales (naive amax-to-scale)
        xs = float(x.detach().abs().amax().clamp_min(1e-6))
        ws = float(self.weight.detach().abs().amax().clamp_min(1e-6))
        gs = float(self.g_scale.item())  # keep previous grad scale
        self.x_scale.fill_(xs)
        self.w_scale.fill_(ws)
        # custom op does FP8 -> BF16 matmul
        y, x_f8, w_f8 = torch.ops.nanogpt.mm(  # type: ignore
            x.contiguous(), self.weight.contiguous(), xs, ws, gs
        )
        if self.bias is not None:
            y = y + self.bias
        return y



TOKENIZER_NAME = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
if tokenizer.pad_token is None:
    eos_token = tokenizer.eos_token
    if eos_token is None:
        raise ValueError(f"Tokenizer {TOKENIZER_NAME!r} has no pad or eos token configured.")
    tokenizer.pad_token = eos_token


# ----------------------------- Utilities -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, D]
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = (x * rms).to(dtype)
        # Return a fresh tensor to avoid potential cudagraph aliasing with torch.compile
        out = x * self.weight
        return out


def swiglu(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # SwiGLU: silu(u) * v
    return F.silu(u) * v


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RoPE:
    """Rotary position embeddings (applied to q, k)."""
    @staticmethod
    def create_cos_sin_cache(seq_len: int, n_elem: int, base: float = 10000.0, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        # n_elem is head_dim
        half = n_elem // 2
        theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        idx_theta = torch.outer(seq_idx, theta)  # [T, half]
        cos = torch.cos(idx_theta).to(dtype)
        sin = torch.sin(idx_theta).to(dtype)
        return cos, sin  # each [T, half]

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]; cos/sin: [T, D/2]
        B, H, T, D = x.shape
        half = D // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos_t = cos[:T].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        sin_t = sin[:T].to(dtype=x.dtype, device=x.device)[None, None, :, :]
        xr1 = x1 * cos_t - x2 * sin_t
        xr2 = x1 * sin_t + x2 * cos_t
        return torch.cat([xr1, xr2], dim=-1)


# ------------------------------ MoE parts ----------------------------

class ExpertSwiGLU(nn.Module):
    """One SwiGLU expert with expansion ~8/3.

    in_features -> hidden (8/3 * d) using two projections for SwiGLU, then down to in_features.
    """
    def __init__(self, d_model: int, bias: bool = False, dropout: float = 0.0, use_fp8: bool = False):
        self.use_fp8 = bool(use_fp8) and _FP8_AVAILABLE
        super().__init__()
        hidden = int(round(d_model * (8.0 / 3.0)))
        # round to multiples of 64 for better kernels
        hidden = (hidden + 63) // 64 * 64
        self.up_u = FP8Linear(d_model, hidden, bias=bias) if self.use_fp8 else nn.Linear(d_model, hidden, bias=bias)
        self.up_v = FP8Linear(d_model, hidden, bias=bias) if self.use_fp8 else nn.Linear(d_model, hidden, bias=bias)
        self.down = FP8Linear(hidden, d_model, bias=bias) if self.use_fp8 else nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up_u(x)
        v = self.up_v(x)
        y = swiglu(u, v)
        y = self.down(y)
        return self.dropout(y)


class Top1Router(nn.Module):
    """Top-1 routing with optional dropless dispatch and load-balancing aux loss.

    When dropless=True: routes all tokens, no capacity truncation.
    When dropless=False: enforces per-expert capacity; extra tokens are dropped (masked to zero contribution).
    """
    def __init__(self, d_model: int, n_experts: int, capacity_factor: float = 1.25, dropless: bool = True,
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0,
                 temperature: float = 1.0, noise_std: float = 0.0, noise_type: str = 'gumbel'):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.dropless = dropless
        self.load_balance_alpha = load_balance_alpha
        self.router_z_loss_coef = router_z_loss_coef
        self.w_gating = nn.Linear(d_model, n_experts, bias=True)
        # routing dynamics
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.noise_type = str(noise_type)

    @torch.no_grad()
    def set_router_state(self, temperature: Optional[float] = None, noise_std: Optional[float] = None):
        if temperature is not None:
            self.temperature = float(temperature)
        if noise_std is not None:
            self.noise_std = float(noise_std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Returns
        - probs: [N, E] softmax probabilities
        - top1_idx: [N] selected expert indices
        - top1_prob: [N] selected expert probabilities
        - aux_loss: scalar tensor for load balancing (and router z-loss if enabled)
        - me: [E] fraction of tokens per expert (no grad)
        - ce: [E] mean gate prob per expert (no grad)
        - entropy_mean: [1] mean token entropy of router softmax (no grad)
        """
        logits = self.w_gating(x)  # [N, E]
        temp = max(1e-5, float(self.temperature))
        logits_base = logits / temp
        probs = F.softmax(logits_base, dim=-1)
        # Noisy top-1 selection for exploration (does not affect aux stats)
        logits_sel = logits_base
        if self.training and self.noise_std > 1e-8:
            if self.noise_type == 'gumbel':
                u = torch.rand_like(logits_sel).clamp_(1e-6, 1 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits_sel = logits_sel + self.noise_std * g
            else:  # gaussian
                logits_sel = logits_sel + self.noise_std * torch.randn_like(logits_sel)
        top1_idx = logits_sel.argmax(dim=-1)
        # Use non-noisy probs to compute selected probability
        top1_prob = probs.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)

        # Load balancing auxiliary loss (Switch-Transformer style)
        # fraction of tokens per expert (me) and mean probability per expert (ce)
        with torch.no_grad():
            N, E = probs.shape
            one_hot_assign = F.one_hot(top1_idx, num_classes=E).float()
            me = one_hot_assign.mean(dim=0)  # [E]
            ce = probs.mean(dim=0)           # [E]
            # Router entropy across tokens
            entropy_mean = (-(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)).mean()
        aux = (self.n_experts * (me * ce).sum())
        aux = self.load_balance_alpha * aux

        # Optional z-loss on router logits (stabilizes softmax)
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits.float(), dim=-1)
            z_loss = (z.square()).mean() * self.router_z_loss_coef
            aux = aux + z_loss.to(aux.dtype)

        return probs, top1_idx, top1_prob, aux, me, ce, entropy_mean


class MoE(nn.Module):
    """Mixture-of-Experts with top-1 routing.

    Implementation emphasizes simplicity for small models and BF16 speed.
    Uses dropless routing by default: every token is processed by its selected expert.
    """
    def __init__(self, d_model: int, n_experts: int, bias: bool, dropout: float,
                 capacity_factor: float = 1.25, dropless: bool = True,
                 load_balance_alpha: float = 0.05, router_z_loss_coef: float = 0.0,
                 router_temperature: float = 1.0, router_noise_std: float = 0.0, router_noise_type: str = 'gumbel',
                 grouped: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.grouped = bool(grouped)
        self.router = Top1Router(
            d_model, n_experts, capacity_factor, dropless,
            load_balance_alpha, router_z_loss_coef,
            temperature=router_temperature, noise_std=router_noise_std, noise_type=router_noise_type,
        )
        self.experts = nn.ModuleList([
            ExpertSwiGLU(d_model, bias=bias, dropout=dropout) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        # compiled expert-apply (lazy)
        self._compiled_apply = None
        self._compiled_apply_grouped = None

        # Fast path when n_experts=1 -> just dense SwiGLU
        self._dense_fallback = (n_experts == 1)
        if self._dense_fallback:
            self.dense = ExpertSwiGLU(d_model, bias=bias, dropout=dropout)

    def _apply_experts_dropless(self, x_flat: torch.Tensor, top1_idx: torch.Tensor, top1_p: torch.Tensor) -> torch.Tensor:
        y_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top1_idx == e)
            if mask.any():
                xe = x_flat[mask]
                ye = self.experts[e](xe)
                if ye.dtype != x_flat.dtype:
                    ye = ye.to(x_flat.dtype)
                ye = ye * top1_p[mask].unsqueeze(-1)
                y_flat[mask] = ye
        return y_flat

    

    def _apply_experts_dropless_grouped(self, x_flat: torch.Tensor, top1_idx: torch.Tensor, top1_p: torch.Tensor) -> torch.Tensor:
        device = x_flat.device
        dtype = x_flat.dtype
        N = x_flat.size(0)
        d = x_flat.size(1)
        # counts per expert and cap
        counts = []
        idx_lists = []
        for e in range(self.n_experts):
            routed = (top1_idx == e)
            if routed.any():
                idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
            else:
                idx_e = torch.empty(0, dtype=torch.long, device=device)
            counts.append(int(idx_e.numel()))
            idx_lists.append(idx_e)
        cap = max(1, max(counts) if counts else 1)
        # pad and stack to [E, cap]
        padded = []
        for idx_e in idx_lists:
            if idx_e.numel() < cap:
                pad = torch.full((cap - idx_e.numel(),), -1, dtype=torch.long, device=device)
                idx_e = torch.cat([idx_e, pad], dim=0)
            else:
                idx_e = idx_e[:cap]
            padded.append(idx_e)
        idx_mat = torch.stack(padded, dim=0)
        valid_mask = (idx_mat >= 0)
        safe_idx = idx_mat.clamp_min(0)
        x_grouped = x_flat[safe_idx]  # [E, cap, d]
        gate_grouped = torch.zeros((self.n_experts, cap, 1), device=device, dtype=dtype)
        top1_p_grouped = top1_p[safe_idx]
        gate_grouped[valid_mask] = top1_p_grouped[valid_mask].unsqueeze(-1).to(dtype)
        # Stack weights and biases
        Wu = torch.stack([exp.up_u.weight for exp in self.experts], dim=0).to(dtype)  # [E, hidden, d]
        WuT = Wu.transpose(1, 2)
        bu = torch.stack([exp.up_u.bias if exp.up_u.bias is not None else torch.zeros(exp.up_u.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
        Wv = torch.stack([exp.up_v.weight for exp in self.experts], dim=0).to(dtype)
        WvT = Wv.transpose(1, 2)
        bv = torch.stack([exp.up_v.bias if exp.up_v.bias is not None else torch.zeros(exp.up_v.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
        WdT = torch.stack([exp.down.weight.t() for exp in self.experts], dim=0).to(dtype)
        bd = torch.stack([exp.down.bias if exp.down.bias is not None else torch.zeros(exp.down.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
        # Batched GEMMs
        U = torch.bmm(x_grouped, WuT) + bu.unsqueeze(1)
        V = torch.bmm(x_grouped, WvT) + bv.unsqueeze(1)
        H = swiglu(U, V)
        Y = torch.bmm(H, WdT) + bd.unsqueeze(1)
        Y = Y * gate_grouped
        Y = Y.masked_fill(~valid_mask.unsqueeze(-1), 0)
        # Scatter back
        y_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            cnt = counts[e]
            if cnt > 0:
                idx_valid = idx_mat[e, :cnt]
                y_flat.index_copy_(0, idx_valid, Y[e, :cnt].to(dtype))
        return y_flat
        # Expert application with top-1 weighting; compiled variant (dynamic=True)
        y_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top1_idx == e)
            if mask.any():
                xe = x_flat[mask]
                ye = self.experts[e](xe)
                # ensure dtype matches destination to avoid index_put dtype mismatch under compile/autocast
                if ye.dtype != x_flat.dtype:
                    ye = ye.to(x_flat.dtype)
                ye = ye * top1_p[mask].unsqueeze(-1)
                y_flat[mask] = ye
        return y_flat

    @_dynamo_disable  # keep routing eager; call into compiled expert compute
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        if self._dense_fallback:
            return self.dense(x), x.new_zeros(())

        # Flatten tokens to route per-token
        x_flat = x.reshape(b * t, d)
        probs, top1_idx, top1_p, aux, me, ce, entropy_mean = self.router(x_flat)

        if self.router.dropless:
            # Dropless: process all tokens by their chosen expert and scatter back
            # choose expert application path
            if self.grouped:
                if self._compiled_apply_grouped is None:
                    try:
                        self._compiled_apply_grouped = torch.compile(
                            self._apply_experts_dropless_grouped, backend="inductor", dynamic=True, fullgraph=False, mode="max-autotune"
                        )
                    except Exception:
                        self._compiled_apply_grouped = self._apply_experts_dropless_grouped
                y_flat = self._compiled_apply_grouped(x_flat, top1_idx, top1_p)
            else:
                if self._compiled_apply is None:
                    try:
                        self._compiled_apply = torch.compile(
                            self._apply_experts_dropless, backend="inductor", dynamic=True, fullgraph=False, mode="max-autotune"
                        )
                    except Exception:
                        self._compiled_apply = self._apply_experts_dropless
                y_flat = self._compiled_apply(x_flat, top1_idx, top1_p)
            y = y_flat.reshape(b, t, d)
            # Stats (no drops in dropless)
            with torch.no_grad():
                N = b * t
                max_frac = me.max()
                num_active = (me > 0).sum()
                drop_frac = torch.zeros((), dtype=torch.float32, device=x.device)
                top1_p_mean = top1_p.mean()
                served = me.clone()  # dropless: all routed tokens are processed
            # Store for logging
            self._last_stats = {
                'aux': aux.detach(),
                'me': me.detach(),
                'served': served.detach(),
                'ce': ce.detach(),
                'entropy_mean': entropy_mean.detach(),
                'max_frac': max_frac.detach(),
                'num_active': num_active.detach(),
                'drop_frac': drop_frac.detach(),
                'top1_p_mean': top1_p_mean.detach(),
                'tokens': torch.tensor(b * t, device=x.device),
            }
            return self.dropout(y), aux
        else:
            # Capacity-limited variant (tokens beyond capacity are dropped to zero contribution)
            N = b * t
            cap = int(math.ceil(self.router.capacity_factor * (N / self.n_experts)))
            if not self.grouped:
                y_flat = torch.zeros_like(x_flat)
                processed = 0
                served_counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.int32)
                for e in range(self.n_experts):
                    # Select up to cap tokens with highest prob to expert e
                    p_e = probs[:, e]
                    routed = (top1_idx == e)
                    if routed.any():
                        idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
                        if idx_e.numel() > cap:
                            pe = p_e[idx_e]
                            topk = torch.topk(pe, cap, dim=0).indices
                            idx_e = idx_e[topk]
                        xe = x_flat[idx_e]
                        ye = self.experts[e](xe)
                        ye = ye * p_e[idx_e].unsqueeze(-1)
                        y_flat[idx_e] = ye
                        n_e = int(idx_e.numel())
                        processed += n_e
                        served_counts[e] = n_e
                y = y_flat.reshape(b, t, d)
                with torch.no_grad():
                    max_frac = me.max()
                    num_active = (me > 0).sum()
                    drop_frac = torch.tensor(1.0 - (processed / max(1, N)), device=x.device)
                    top1_p_mean = top1_p.mean()
                    served = served_counts.to(torch.float32) / max(1, N)
                self._last_stats = {
                    'aux': aux.detach(),
                    'me': me.detach(),
                    'served': served.detach(),
                    'ce': ce.detach(),
                    'entropy_mean': entropy_mean.detach(),
                    'max_frac': max_frac.detach(),
                    'num_active': num_active.detach(),
                    'drop_frac': drop_frac.detach(),
                    'top1_p_mean': top1_p_mean.detach(),
                    'tokens': torch.tensor(b * t, device=x.device),
                    'cap': torch.tensor(cap, device=x.device),
                }
                return self.dropout(y), aux
            else:
                # Grouped/padded dispatch with batched GEMMs across experts
                device = x_flat.device
                dtype = x_flat.dtype
                d_model = d
                # Prepare per-expert indices up to capacity
                idx_lists = []
                counts = []
                processed = 0
                for e in range(self.n_experts):
                    p_e = probs[:, e]
                    routed = (top1_idx == e)
                    if routed.any():
                        idx_e = torch.nonzero(routed, as_tuple=False).squeeze(-1)
                        if idx_e.numel() > cap:
                            pe = p_e[idx_e]
                            topk = torch.topk(pe, cap, dim=0).indices
                            idx_e = idx_e[topk]
                    else:
                        idx_e = torch.empty(0, dtype=torch.long, device=device)
                    counts.append(int(idx_e.numel()))
                    processed += int(idx_e.numel())
                    # pad to cap
                    if idx_e.numel() < cap:
                        pad = torch.full((cap - idx_e.numel(),), -1, dtype=torch.long, device=device)
                        idx_e = torch.cat([idx_e, pad], dim=0)
                    else:
                        idx_e = idx_e[:cap]
                    idx_lists.append(idx_e)

                idx_mat = torch.stack(idx_lists, dim=0)  # [E, cap]
                valid_mask = (idx_mat >= 0)
                # Gather tokens with padding
                safe_idx = idx_mat.clamp_min(0)
                x_grouped = x_flat[safe_idx]            # [E, cap, d]
                gate_grouped = torch.zeros((self.n_experts, cap, 1), device=device, dtype=dtype)
                # selected prob per token for the chosen expert
                top1_p_grouped = top1_p[safe_idx]
                gate_grouped[valid_mask] = top1_p_grouped[valid_mask].unsqueeze(-1).to(dtype)

                # Stack weights for grouped GEMMs
                # up_u
                Wu = torch.stack([exp.up_u.weight for exp in self.experts], dim=0).to(dtype)  # [E, hidden, d]
                WuT = Wu.transpose(1, 2)  # [E, d, hidden]
                bu = torch.stack([exp.up_u.bias if exp.up_u.bias is not None else torch.zeros(exp.up_u.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
                # up_v
                Wv = torch.stack([exp.up_v.weight for exp in self.experts], dim=0).to(dtype)
                WvT = Wv.transpose(1, 2)
                bv = torch.stack([exp.up_v.bias if exp.up_v.bias is not None else torch.zeros(exp.up_v.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)
                # down
                WdT = torch.stack([exp.down.weight.t() for exp in self.experts], dim=0).to(dtype)  # [E, hidden, d]
                bd = torch.stack([exp.down.bias if exp.down.bias is not None else torch.zeros(exp.down.out_features, device=device, dtype=dtype) for exp in self.experts], dim=0).to(dtype)

                # Batched GEMMs
                U = torch.bmm(x_grouped, WuT) + bu.unsqueeze(1)  # [E, cap, hidden]
                V = torch.bmm(x_grouped, WvT) + bv.unsqueeze(1)
                H = swiglu(U, V)
                Y = torch.bmm(H, WdT) + bd.unsqueeze(1)  # [E, cap, d]
                Y = Y * gate_grouped
                # Zero-out padding rows
                Y = Y.masked_fill(~valid_mask.unsqueeze(-1), 0)

                # Scatter back
                y_flat = torch.zeros_like(x_flat)
                for e in range(self.n_experts):
                    cnt = counts[e]
                    if cnt > 0:
                        idx_valid = idx_mat[e, :cnt]
                        y_flat.index_copy_(0, idx_valid, Y[e, :cnt])

                y = y_flat.reshape(b, t, d)
                with torch.no_grad():
                    max_frac = me.max()
                    num_active = (me > 0).sum()
                    drop_frac = torch.tensor(1.0 - (processed / max(1, N)), device=x.device)
                    top1_p_mean = top1_p.mean()
                    served = torch.tensor(counts, device=x.device, dtype=torch.float32) / max(1, N)
                self._last_stats = {
                    'aux': aux.detach(),
                    'me': me.detach(),
                    'served': served.detach(),
                    'ce': ce.detach(),
                    'entropy_mean': entropy_mean.detach(),
                    'max_frac': max_frac.detach(),
                    'num_active': num_active.detach(),
                    'drop_frac': drop_frac.detach(),
                    'top1_p_mean': top1_p_mean.detach(),
                    'tokens': torch.tensor(b * t, device=x.device),
                    'cap': torch.tensor(cap, device=x.device),
                }
                return self.dropout(y), aux


# --------------------------- Transformer blocks ---------------------------
# (Only GQA/GatedGQA variants retained)
#
# MultiheadAttention and GatedMultiheadAttention have been removed. If n_kv_heads == n_head,
# GQA degenerates to standard MHA (group_size=1), so this keeps behavior while simplifying the code.
class GQA(nn.Module):
    """Grouped-Query Attention: Q has n_head, K/V have n_kv_heads, n_head % n_kv_heads == 0.
    Replicates K/V across groups to match Q heads, then uses SDPA.
    """
    def __init__(self, d_model: int, n_head: int, n_kv_heads: int, bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6, use_fp8: bool = False):
        self.use_fp8 = bool(use_fp8)
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_heads == 0
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        self.group_size = n_head // n_kv_heads
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = float(dropout)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        hk = self.n_kv_heads
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)      # [b,h,t,dh]
        k = self.k_proj(x).view(b, t, hk, -1).transpose(1, 2)     # [b,hk,t,dh]
        v = self.v_proj(x).view(b, t, hk, -1).transpose(1, 2)     # [b,hk,t,dh]
        # QK-Norm before RoPE (apply to q and k; k per kv-head)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_cache is not None:
            cos, sin = rope_cache
            q = RoPE.apply_rope(q, cos, sin)
            k = RoPE.apply_rope(k, cos, sin)
        # replicate kv along head groups
        if hk != h:
            k = k.repeat_interleave(self.group_size, dim=1)  # [b,h,t,dh]
            v = v.repeat_interleave(self.group_size, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.o_proj(y)
        return y


class GatedGQA(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_heads: int, bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6, use_fp8: bool = False):
        self.use_fp8 = bool(use_fp8)
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_heads == 0
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        self.group_size = n_head // n_kv_heads
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gate_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = float(dropout)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_head
        hk = self.n_kv_heads
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)
        k = self.k_proj(x).view(b, t, hk, -1).transpose(1, 2)
        v = self.v_proj(x).view(b, t, hk, -1).transpose(1, 2)
        # QK-Norm before RoPE
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_cache is not None:
            cos, sin = rope_cache
            q = RoPE.apply_rope(q, cos, sin)
            k = RoPE.apply_rope(k, cos, sin)
        if hk != h:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        gate = torch.sigmoid(self.gate_proj(x)).view(b, t, h, -1).transpose(1, 2)
        y = y * gate
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.o_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_heads: int, bias: bool, dropout: float,
                 n_experts: int, capacity_factor: float, dropless: bool,
                 load_balance_alpha: float, router_z_loss_coef: float,
                 attn_gate: str = 'none', use_rope: bool = True,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 use_compile: bool = False, use_fp8: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.use_rope = use_rope
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        # Always use GQA variants. When n_kv_heads == n_head, this reduces to standard MHA.
        if attn_gate == 'sigmoid_head':
            self.attn = GatedGQA(d_model, n_head, n_kv_heads, bias=bias, dropout=dropout,
                                 qk_norm=qk_norm, qk_norm_eps=qk_norm_eps, use_fp8=use_fp8)
        else:
            self.attn = GQA(d_model, n_head, n_kv_heads, bias=bias, dropout=dropout,
                            qk_norm=qk_norm, qk_norm_eps=qk_norm_eps, use_fp8=use_fp8)
        self.ln2 = RMSNorm(d_model)
        self.moe = MoE(
            d_model, n_experts, bias=bias, dropout=dropout,
            capacity_factor=capacity_factor, dropless=dropless,
            load_balance_alpha=load_balance_alpha, router_z_loss_coef=router_z_loss_coef,
            router_temperature=1.0, router_noise_std=0.0, router_noise_type='gumbel',
            grouped=False,
        )
        # compile attention submodule (shape-stable)
        if use_compile:
            try:
                self.attn = torch.compile(self.attn, backend="inductor", dynamic=True, fullgraph=False, mode="max-autotune")
            except Exception:
                pass

    def forward(self, x: torch.Tensor, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.ln1(x)
        attn_out = self.attn(attn_in, rope_cache if self.use_rope else None)
        x = x + attn_out
        y, aux = self.moe(self.ln2(x))
        x = x + y
        return x, aux


class TinyMoETransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 32768,
                 n_layer: int = 10,
                 n_head: int = 8,
                 n_kv_heads: Optional[int] = None,
                 d_model: int = 512,
                 block_size: int = 2048,
                 dropout: float = 0.0,
                 bias: bool = False,
                 n_experts: int = 4,
                 capacity_factor: float = 1.25,
                 dropless: bool = True,
                 load_balance_alpha: float = 0.05,
                 router_z_loss_coef: float = 0.0,
                 attn_gate: str = 'none',
                 router_temperature: float = 1.0,
                 router_noise_std: float = 0.0,
                 router_noise_type: str = 'gumbel',
                 use_rope: bool = True,
                 rope_theta: float = 10000.0,
                 moe_grouped: bool = False,
                 qk_norm: bool = False,
                 qk_norm_eps: float = 1e-6,
                 compile_submodules: bool = False,
                 fp8: bool = False,
                 ):  # noqa: E501
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.use_rope = use_rope
        self.head_dim = d_model // n_head
        self.n_head = n_head
        self.n_kv_heads = n_head if n_kv_heads is None else int(n_kv_heads)
        assert self.n_head % self.n_kv_heads == 0, "n_head must be a multiple of n_kv_heads"
        if self.use_rope:
            assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE"

        self.wte = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([
            TransformerBlock(
                d_model, n_head, self.n_kv_heads, bias, dropout,
                n_experts, capacity_factor, dropless,
                load_balance_alpha, router_z_loss_coef,
                attn_gate=attn_gate,
                use_rope=use_rope,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                use_compile=compile_submodules,
                use_fp8=fp8,
            )
            for _ in range(n_layer)
        ])
        # set grouped flag into each block's MoE
        if moe_grouped:
            for blk in self.h:
                if hasattr(blk, 'moe'):
                    blk.moe.grouped = True
        # initialize router dynamics state across blocks
        for blk in self.h:
            if hasattr(blk, 'moe'):
                blk.moe.router.set_router_state(router_temperature, router_noise_std)
                blk.moe.router.noise_type = router_noise_type
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.wte.weight

        # RoPE cache stored at model level
        if self.use_rope:
            cos, sin = RoPE.create_cos_sin_cache(self.block_size, self.head_dim, base=rope_theta)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        assert t <= self.block_size
        x = self.wte(idx)
        x = self.drop(x)
        aux_losses = []
        rope_cache = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for blk in self.h:
            x, aux = blk(x, rope_cache=rope_cache)
            aux_losses.append(aux)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # compute CE in float32 for stability
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1)
            # sum of aux losses (already scaled by alpha)
            if aux_losses:
                aux_total = torch.stack([a.float() for a in aux_losses]).mean()
            else:
                aux_total = logits.new_zeros(())
            loss = ce + aux_total
        return logits, loss

    @torch.no_grad()
    def num_parameters(self) -> int:
        return count_parameters(self)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
