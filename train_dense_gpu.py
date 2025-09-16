import os
import time, math, argparse, statistics as stats, itertools, csv, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Optional
# Delay setting TORCH_LOGS until after imports to avoid logging initialization issues
import torch
from torch.utils.data import DataLoader
# Now safely set TORCH_LOGS to keep logs quiet
os.environ.setdefault("TORCH_LOGS", "")

# -------------------------
# Dataset alias resolver
# -------------------------
def resolve_dataset_and_config(name: str, cfg: Optional[str]) -> Tuple[str, Optional[str]]:
    """Map convenient short names to repo datasets + default configs."""
    if not name:
        return name, cfg
    key = name.lower()
    # aliases
    if key in {"owt", "openwebtext"}:
        return "Skylion007/openwebtext", None
    if key in {"c4", "allenai/c4"}:
        return "allenai/c4", (cfg or "en")
    if key in {"fineweb", "fw"}:
        return "HuggingFaceFW/fineweb", cfg  # pick a subset in cfg if desired
    if key in {"fineweb-edu", "fw-edu"}:
        return "HuggingFaceFW/fineweb-edu", cfg
    if key in {"wikitext", "wt103", "wikitext103"}:
        return "wikitext", (cfg or "wikitext-103-raw-v1")
    return name, cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
except Exception:
    AutoTokenizer = None  # only needed for HF dataset mode

# -------------------------
# Utilities & defaults
# -------------------------
def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def human(n: int) -> str:
    return f"{n/1e6:.1f}M"

def set_cuda_env():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Prefer Flash / MemEff SDPA where possible
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# -------------------------
# Tiny dense transformer LM (GPU) + RMSNorm/QK-Norm/GQA/SwiGLU
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_heads: int,
                 qk_norm: bool = True, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_heads == 0, "n_head must be a multiple of n_kv_heads (GQA)"
        self.n_head = n_head
        self.n_kv = n_kv_heads
        self.head_dim = d_model // n_head
        # merged QKV for GQA: (n_head + 2*n_kv) * head_dim
        qkv_out = (n_head + 2 * n_kv_heads) * self.head_dim
        self.qkv = nn.Linear(d_model, qkv_out, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_scale = nn.Parameter(torch.ones(self.head_dim))
            self.k_scale = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv, [self.n_head*self.head_dim, self.n_kv*self.head_dim, self.n_kv*self.head_dim], dim=-1
        )
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv,   self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            q = q * self.q_scale
            k = k * self.k_scale
        if self.n_kv != self.n_head:
            repeat = self.n_head // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, d_model, pdrop=0.0):
        super().__init__()
        hidden = int((8.0/3.0) * d_model)
        self.up_g = nn.Linear(d_model, hidden, bias=False)
        self.up_f = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(pdrop)
    def forward(self, x):
        g = F.silu(self.up_g(x))
        f = self.up_f(x)
        return self.drop(self.down(g * f))

class Block(nn.Module):
    def __init__(self, d_model, n_head, n_kv_heads, qk_norm=True,
                 attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0, checkpoint=False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_kv_heads, qk_norm, attn_pdrop, resid_pdrop)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, pdrop=mlp_pdrop)
        self.checkpoint = checkpoint

    def forward(self, x):
        if self.checkpoint and self.training:
            def fn1(x): return x + self.attn(self.ln1(x))
            def fn2(x): return x + self.mlp(self.ln2(x))
            x = torch.utils.checkpoint.checkpoint(fn1, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(fn2, x, use_reentrant=False)
            return x
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        n_layer=12,
        n_head=11,
        n_kv_heads=11,
        d_model=704,
        block_size=1024,
        pdrop=0.1,
        checkpoint=False,
        qk_norm=True,
    ):
        super().__init__()
        assert d_model % n_head == 0
        assert n_head % n_kv_heads == 0
        self.vocab_size, self.block_size = vocab_size, block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(pdrop)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, n_kv_heads, qk_norm, pdrop, pdrop, pdrop, checkpoint=checkpoint)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} > block_size={self.block_size}")
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

# -------------------------
# Data pipelines
# -------------------------
@dataclass
class DataGen:
    """Synthetic Zipfian token stream (fallback)."""
    vocab_size: int; block_size: int; batch_size: int; seed: int = 0
    def __post_init__(self):
        self.gen = torch.Generator(device="cpu").manual_seed(self.seed)
        ranks = torch.arange(1, self.vocab_size+1, dtype=torch.float64)
        probs = (1.0 / ranks).double(); probs = (probs / probs.sum()).float()
        self.probs = probs
    def next_batch(self, device="cuda"):
        x = torch.multinomial(self.probs, num_samples=self.batch_size * self.block_size,
                              replacement=True, generator=self.gen)
        x = x.view(self.batch_size, self.block_size).to(device, non_blocking=True)
        y = torch.roll(x, shifts=-1, dims=1)
        return x, y

class HFStreamPacker(torch.utils.data.IterableDataset):
    """
    Stream text with datasets.load_dataset(..., streaming=True),
    tokenize with a HF tokenizer, then pack into fixed-length (block_size+1) windows.
    Yields LongTensors of shape [block_size+1].
    """
    def __init__(self, ds_name: str, ds_config: Optional[str], split: str,
                 tokenizer, block_size: int, seed: int = 1234, shuffle_buffer: int = 1_000):
        super().__init__()
        self.ds_name, self.ds_config, self.split = ds_name, ds_config, split
        self.tok = tokenizer
        self.block = block_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        ds = load_dataset(self.ds_name, self.ds_config, split=self.split, streaming=True)
        try:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        except Exception:
            pass  # some datasets don't support shuffle in streaming mode
        buf: List[int] = []
        for ex in ds:
            # pick the right text field
            if isinstance(ex, dict):
                txt = ex.get("text") or ex.get("content") or ex.get("document") or ""
            else:
                txt = str(ex)
            ids = self.tok.encode(txt, add_special_tokens=False)
            # append EOS after each doc if tokenizer has one
            if getattr(self.tok, "eos_token_id", None) is not None:
                ids.append(self.tok.eos_token_id)
            buf.extend(ids)
            # pack windows
            L = self.block + 1
            while len(buf) >= L:
                chunk = torch.tensor(buf[:L], dtype=torch.long)
                del buf[:self.block]  # advance by block (overlap of 1 for next-token targets)
                yield chunk

class LoaderBatcher:
    """Wrap a DataLoader over HFStreamPacker to provide .next_batch(device)."""
    def __init__(self, loader: torch.utils.data.DataLoader, block_size: int):
        self.loader = loader
        self.it = None
        # expose for accounting
        self.block_size = int(block_size)
        # if DataLoader has a static batch_size, use it (helpful for logging)
        self.batch_size = int(loader.batch_size) if loader.batch_size is not None else None
    def _next_tensor(self):
        if self.it is None:
            self.it = iter(self.loader)
        try:
            batch = next(self.it)  # [B, L]
        except StopIteration:
            self.it = iter(self.loader)
            batch = next(self.it)
        return batch
    def next_batch(self, device="cuda"):
        batch = self._next_tensor()  # [B, block+1]
        x = batch[:, :-1].contiguous().to(device, non_blocking=True)
        y = batch[:, 1: ].contiguous().to(device, non_blocking=True)
        return x, y

@torch.inference_mode()
def eval_stream_ppl(model, batcher: LoaderBatcher, iters: int, amp_dtype: torch.dtype):
    """Quick validation on a streaming loader; returns (mean_loss, ppl)."""
    losses = []
    for _ in range(iters):
        x, y = batcher.next_batch(device=next(model.parameters()).device.type)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(float(loss))
    m = sum(losses)/len(losses)
    try:
        ppl = math.exp(m)
    except OverflowError:
        ppl = float("inf")
    return m, ppl
# -------------------------
# Optimizers (local muon/sophia)
# -------------------------
def split_params_for_muon(model) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    hidden, aux = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("tok_emb" in name) or ("pos_emb" in name) or ("lm_head" in name):
            aux.append(p)
        elif p.ndim >= 2:
            hidden.append(p)
        else:
            aux.append(p)
    return hidden, aux

class MuonSophiaSingleDevice:
    """Muon on hidden params, SophiaG on aux params (no DDP)."""
    def __init__(self, hidden, aux, muon_kw, sophia_kw):
        from muon import SingleDeviceMuon
        from sophia import SophiaG
        self.muon = SingleDeviceMuon(hidden, **muon_kw)
        self.sophia = SophiaG(aux, **sophia_kw)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.sophia.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, bs: int):
        self.muon.step()
        self.sophia.step(bs=bs)

    @torch.no_grad()
    def update_hessian(self):
        self.sophia.update_hessian()

def build_optimizer(kind: str, model: nn.Module, foreach=True, fused=True,
                    muon_lr=3e-2, muon_wd=1e-2,
                    sophia_lr=1e-3, sophia_b1=0.965, sophia_b2=0.99, sophia_rho=0.1, sophia_wd=0.2):
    """Muon + Sophia only (composite)."""
    kind = kind.lower()
    if kind not in {"muon_sophia", "muon+sophia", "ms"}:
        raise ValueError("This build only supports Muon+Sophia. Use --opt-list muon_sophia")
    hidden, aux = split_params_for_muon(model)
    opt = MuonSophiaSingleDevice(
        hidden, aux,
        muon_kw=dict(lr=muon_lr, weight_decay=muon_wd, momentum=0.95),
        sophia_kw=dict(lr=sophia_lr, betas=(sophia_b1, sophia_b2), rho=sophia_rho, weight_decay=sophia_wd),
    )
    return opt, dict(k_hess=10)

# -------------------------
# Train loop (GPU, BF16 AMP, compile)
# -------------------------
@dataclass
class RunResult:
    name: str
    mean_ms: float
    tok_s: float
    final_loss: float

def train_once(
    name: str,
    model: nn.Module,
    data: DataGen,
    steps: int,
    device: str,
    optimizer_kind: str,
    foreach: bool,
    fused: bool,
    use_compile: bool,
    dynamic_shapes: bool,
    grad_accum: int,
    amp_dtype: torch.dtype,
    k_hess_override: Optional[int],
    log_every: int=50,
    k_hess_after: Optional[int]=None,
    k_hess_switch_step: int=0,
    val_every: int=0,
    val_iters: int=50,
    val_batcher: Optional[LoaderBatcher]=None,
    clip_val: float = 0.0,
) -> RunResult:
    scaler = None  # bf16: no scaler needed; fp16: enable GradScaler
    if amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    opt, sophia_cfg = build_optimizer(
        optimizer_kind, model, foreach=foreach, fused=fused
    )
    if sophia_cfg and k_hess_override:
        sophia_cfg["k_hess"] = k_hess_override

    model = model.to(device, non_blocking=True)
    if use_compile:
        model = torch.compile(model, backend="inductor", dynamic=dynamic_shapes)

    times=[]; losses=[]; iter_num=0
    # Warmup & trigger compile
    x,y = data.next_batch(device)
    model.train(); opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=amp_dtype):
        _, loss = model(x, y)
    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    # warmup optimizer step (Sophia-style opts require bs)
    # infer tokens/iter directly from the warmup batch shape
    bsz, seq = x.size(0), x.size(1)
    bs_warmup = bsz * seq
    if sophia_cfg is None:
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
    else:
        if scaler:
            scaler.unscale_(opt)
            opt.step(bs=bs_warmup)
            scaler.update()
        else:
            opt.step(bs=bs_warmup)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    toks_per_it = bsz * seq * grad_accum
    while iter_num < steps:
        t0 = time.perf_counter()
        # bsz, seq may change if you alter data settings; keep tokens/iter accurate
        # (we already have bsz/seq from warmup; if you want to re-infer every iter,
        #  uncomment the next two lines after fetching x,y)
        # bsz, seq = x.size(0), x.size(1)
        total_loss = 0.0
        for micro in range(grad_accum):
            x,y = data.next_batch(device)
            model.train()
            # only zero at the start of an accumulation window
            if micro == 0: opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                _, loss = model(x, y)
            total_loss += float(loss.detach())
            if scaler:
                scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()
        avg_loss = total_loss / max(1, grad_accum)
        # step
        bs = toks_per_it
        if sophia_cfg is None:
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
        else:
            if scaler:
                scaler.unscale_(opt)  # sophia expects real grads; but usually fine without
            # optional grad clip
            if clip_val and clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            opt.step(bs=bs)
            k = sophia_cfg.get("k_hess", 10)
            if k_hess_after and (iter_num + 1) >= k_hess_switch_step:
                k = k_hess_after
            if (iter_num + 1) % k == 0:
                # extra sampled pass for Hessian EMA
                x,y = data.next_batch(device)
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits, _ = model(x, None)
                    with torch.no_grad():
                        y_samp = torch.distributions.Categorical(logits=logits).sample()
                    loss_s = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y_samp.reshape(-1)
                    )
                if scaler:
                    scaler.scale(loss_s).backward()
                    scaler.unscale_(opt)
                else:
                    loss_s.backward()
                opt.update_hessian()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        losses.append(avg_loss)
        iter_num += 1
        do_val = val_every and val_batcher is not None and (iter_num % val_every == 0)
        if do_val:
            model.eval()
            vl, vp = eval_stream_ppl(model, val_batcher, iters=val_iters, amp_dtype=amp_dtype)
            model.train()
            print(f"[{name:12s}]   (val) loss {vl:.4f}  ppl {vp:.1f}")
        if (iter_num % log_every == 0) or (iter_num == steps):
            print(f"[{name:12s}] step {iter_num:4d}/{steps}  "
                  f"loss {losses[-1]:.4f}  avg {1e3*stats.mean(times):6.2f} ms/it")

    mean_s = stats.mean(times)
    return RunResult(
        name=name,
        mean_ms=mean_s*1e3,
        tok_s=toks_per_it/mean_s,
        final_loss=losses[-1],
    )

def main():
    ap = argparse.ArgumentParser()
    # model
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=11)       # 704/11=64
    ap.add_argument("--d-model", type=int, default=704)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--pdrop", type=float, default=0.1)
    ap.add_argument("--ckpt", action="store_true", default=False)
    ap.add_argument("--qk-norm", action="store_true", default=True)
    # data
    ap.add_argument("--batch", type=int, default=8)        # micro-batch
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grad-accum", type=int, default=1)   # increase for larger effective batch
    # optimizers
    ap.add_argument("--opt-list", type=str, default="muon_sophia", help="This build only supports muon_sophia")
    ap.add_argument("--foreach", action="store_true", default=True, help="Use foreach AdamW. If --fused is set, foreach is auto-disabled.")
    ap.add_argument("--fused", action="store_true", default=True, help="Use fused AdamW on GPU. Incompatible with --foreach.")
    ap.add_argument("--k-hess", type=int, default=10)
    ap.add_argument("--k-hess-after", type=int, default=0, help="0=disabled; use this k after switch")
    ap.add_argument("--k-hess-switch-step", type=int, default=0, help="step to switch cadence")
    ap.add_argument("--muon-lr", type=float, default=3e-2)
    ap.add_argument("--muon-wd", type=float, default=1e-2)
    ap.add_argument("--sophia-lr", type=float, default=1e-3)
    ap.add_argument("--sophia-b1", type=float, default=0.965)
    ap.add_argument("--sophia-b2", type=float, default=0.99)
    ap.add_argument("--sophia-rho", type=float, default=0.1)
    ap.add_argument("--sophia-wd", type=float, default=0.2)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    # data/tokenizer (HF)
    ap.add_argument("--dataset", type=str, default="", help="HF dataset name, e.g. 'wikitext' or 'openwebtext'; empty = synthetic")
    ap.add_argument("--dataset-config", type=str, default="",
                    help="HF dataset config, if any (aliases fill sensible defaults)")
    ap.add_argument("--split", type=str, default="train", help="split name for training")
    ap.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name or local path")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--shuffle-buffer", type=int, default=1000, help="approx shuffle buffer for streaming")
    ap.add_argument("--eos-as-pad", action="store_true", default=True, help="set pad_token=eos if tokenizer has no pad")
    ap.add_argument("--synthetic-vocab", type=int, default=32000, help="vocab for synthetic mode")
    ap.add_argument("--prefetch-factor", type=int, default=8)
    # validation + logging
    ap.add_argument("--val-split", type=str, default="validation")
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=0, help="evaluate every N steps (0=off)")
    ap.add_argument("--csv", type=str, default="", help="optional CSV log path")
    # compile/amp
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--dynamic", action="store_true", default=False)
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    set_cuda_env()
    device = "cuda"

    # dtype selection
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    # tokenizer / dataset
    use_hf_data = bool(args.dataset)
    val_batcher = None
    if use_hf_data:
        # map short names (e.g. "openwebtext" -> "Skylion007/openwebtext")
        ds_name, ds_config = resolve_dataset_and_config(args.dataset, args.dataset_config if args.dataset_config else None)
        if AutoTokenizer is None:
            raise SystemExit("Please `pip install transformers datasets` for HF dataset mode.")
        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        # ensure eos & pad
        if tok.eos_token is None:
            tok.add_special_tokens({"eos_token": "</s>"})
        if args.eos_as_pad and (tok.pad_token is None):
            tok.pad_token = tok.eos_token
        # pad tokenizer vocab to a multiple of 128 for kernel alignment (testing speedup)
        _m = 128
        _v0 = int(len(tok))  # include any already-added tokens
        _target = ((_v0 + _m - 1) // _m) * _m
        _n_add = _target - _v0
        if _n_add > 0:
            tok.add_special_tokens({"additional_special_tokens": [f"<extra_pad_{i}>" for i in range(_n_add)]})
            # keep pad token if requested
            if args.eos_as_pad and (tok.pad_token is None):
                tok.pad_token = tok.eos_token
            print(f"Padded tokenizer vocab from {_v0} to {len(tok)} (x{_m}).")
        # update vocab size to tokenizer size (possibly padded)
        args.vocab = int(len(tok))
        # build streaming dataset and loader
        stream = HFStreamPacker(
            ds_name=ds_name,
            ds_config=ds_config,
            split=args.split,
            tokenizer=tok,
            block_size=args.block_size,
            seed=1234,
            shuffle_buffer=args.shuffle_buffer,
        )
        loader = torch.utils.data.DataLoader(
            stream, batch_size=args.batch, num_workers=args.num_workers,
            pin_memory=True, drop_last=True, collate_fn=lambda x: torch.stack(x, dim=0),
            persistent_workers=True, prefetch_factor=args.prefetch_factor
        )
        batcher_factory = lambda: LoaderBatcher(loader, args.block_size)
        # validation loader (optional; ignore failures gracefully)
        try:
            vname, vcfg = resolve_dataset_and_config(args.dataset, args.dataset_config if args.dataset_config else None)
            vs = HFStreamPacker(vname, vcfg, split=args.val_split, tokenizer=tok,
                                block_size=args.block_size, seed=4321, shuffle_buffer=args.shuffle_buffer//2 or 1000)
            vloader = torch.utils.data.DataLoader(
                vs, batch_size=args.batch, num_workers=max(1, args.num_workers//2),
                pin_memory=True, drop_last=True, collate_fn=lambda x: torch.stack(x, dim=0),
                persistent_workers=True, prefetch_factor=max(2, args.prefetch_factor//2)
            )
            val_batcher = LoaderBatcher(vloader, args.block_size)
        except Exception as _:
            val_batcher = None
    else:
        # synthetic fallback
        if args.vocab is None or args.vocab <= 0:
            args.vocab = args.synthetic_vocab
        batcher_factory = lambda: DataGen(args.vocab, args.block_size, args.batch, seed=1234)
        val_batcher = None

    # build base model & report params (init weights once; copy per optimizer)
    model = TinyTransformerLM(
        vocab_size=args.vocab,
        n_layer=args.layers,
        n_head=args.heads,
        n_kv_heads=max(1, args.heads // 4),
        d_model=args.d_model,
        block_size=args.block_size,
        pdrop=args.pdrop,
        checkpoint=args.ckpt,
        qk_norm=args.qk_norm,
    )
    # resize embeddings if tokenizer added new tokens (rare if using pretrained tokenizer path)
    # (weight tying is already in place)
    if use_hf_data:
        emb = model.tok_emb
        if emb.num_embeddings != args.vocab:
            new = nn.Embedding(args.vocab, emb.embedding_dim, device=emb.weight.device, dtype=emb.weight.dtype)
            with torch.no_grad():
                n = min(emb.num_embeddings, args.vocab)
                new.weight[:n].copy_(emb.weight[:n])
            model.tok_emb = new
            model.lm_head.weight = model.tok_emb.weight

    total = n_params(model)
    print(f"Model params: {human(total)}  "
          f"(V={args.vocab}, L={args.layers}, H={args.heads}, D={args.d_model})")

    # sanity: ~100M target hint
    # d_model=704, heads=11, layers=12, vocab=32k -> ~94M params (no extra lm_head due to tying)

    # run each optimizer on fresh copy (same init) + fresh data stream
    state_dict = model.state_dict()
    results: List[RunResult] = []
    csvw = None
    f = None
    if args.csv:
        path = pathlib.Path(args.csv)
        first = not path.exists()
        f = path.open("a", newline="")
        csvw = csv.writer(f)
        if first:
            csvw.writerow(["opt", "step", "train_loss", "ms_per_it", "val_loss", "val_ppl"])
    for opt_name in [s.strip() for s in args.opt_list.split(",") if s.strip()]:
        print(f"\n=== {opt_name} ===")
        # fresh model and fresh data stream (same seed for fairness)
        m = TinyTransformerLM(
            vocab_size=args.vocab, n_layer=args.layers, n_head=args.heads,
            n_kv_heads=max(1, args.heads // 4), d_model=args.d_model,
            block_size=args.block_size, pdrop=args.pdrop, checkpoint=args.ckpt,
            qk_norm=args.qk_norm
        )
        m.load_state_dict(state_dict)  # same init for fair comparison
        data = batcher_factory()  # fresh iterator / generator per optimizer
        r = train_once(
            name=opt_name,
            model=m,
            data=data,
            steps=args.steps,
            device=device,
            optimizer_kind=opt_name,
            foreach=args.foreach,
            fused=args.fused,
            use_compile=args.compile,
            dynamic_shapes=args.dynamic,
            grad_accum=args.grad_accum,
            amp_dtype=amp_dtype,
            k_hess_override=args.k_hess,
            k_hess_after=(args.k_hess_after or None),
            k_hess_switch_step=args.k_hess_switch_step,
            log_every=max(20, args.steps//5),
            val_every=args.val_every,
            val_iters=args.val_iters,
            val_batcher=val_batcher,
            clip_val=args.clip_grad,
        )
        results.append(r)
        if csvw:
            csvw.writerow([opt_name, args.steps, f"{r.final_loss:.6f}", f"{r.mean_ms:.3f}", "", ""])
            f.flush()

    if f:
        f.close()

    # summary
    print("\n=== Summary (GPU) ===")
    w = max(len(r.name) for r in results)
    for r in results:
        print(f"{r.name:<{w}}  {r.mean_ms:7.2f} ms/it   {r.tok_s:9.0f} tok/s   loss {r.final_loss:.4f}")

if __name__ == "__main__":
    os.environ.setdefault("TORCH_LOGS","")   # quieter
    main()















