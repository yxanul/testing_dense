#!/usr/bin/env python3
"""
Training script for TinyMoETransformer (BF16) with HF streaming data and MoE router metrics.

Adds robust checkpointing:
- Save 'last' every eval
- Save 'best' when val improves
- Optional periodic saves via --save_interval (e.g., 500,1000,...)
- Resume from checkpoint (--resume or --auto_resume)
"""

import math
import time
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

try:
    from wandb_logger import WandBLogger
except Exception:  # pragma: no cover
    class WandBLogger:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def log_metrics(self, *args, **kwargs): pass
        def log_eval(self, *args, **kwargs): pass
        def set_summary(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass

from model_experimental import TinyMoETransformer, tokenizer


# --------------------------- Data + training ---------------------------

@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 32768
    n_layer: int = 10
    n_head: int = 8
    n_kv_heads: Optional[int] = None
    d_model: int = 512
    n_experts: int = 4
    block_size: int = 2048
    dropout: float = 0.0
    bias: bool = False
    capacity_factor: float = 1.25
    dropless: bool = True
    load_balance_alpha: float = 0.05
    router_z_loss_coef: float = 0.0
    attn_gate: str = 'none'  # 'none' or 'sigmoid_head'
    use_rope: bool = True
    rope_theta: float = 10000.0
    # QK-Norm
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    # Router dynamics
    router_temp_init: float = 1.5
    router_temp_final: float = 1.0
    router_temp_anneal_iters: int = 1000
    router_noise_std_init: float = 0.5
    router_noise_decay_iters: int = 1000
    router_noise_type: str = 'gumbel'  # 'gumbel' or 'gaussian'
    moe_grouped: bool = False

    # Training
    device: str = 'cuda'
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    grad_clip: float = 1.0
    compile: bool = False
    seed: int = 1337
    fp8: bool = False

    # Data
    dataset_name: str = 'HuggingFaceFW/fineweb-edu'
    dataset_config: str = 'sample-10BT'
    text_key: str = 'text'
    shuffle_buffer: int = 8192
    eval_take: int = 512

    # IO
    checkpoint_dir: str = 'checkpoints_moe_bf16'
    log_interval: int = 10
    save_interval: int = 0          # set >0 to save periodic checkpoints
    keep_checkpoints: int = 3       # max periodic checkpoints to retain (0=keep all)
    resume: Optional[str] = None    # path to checkpoint file or dir
    auto_resume: bool = False       # auto-resume from last in checkpoint_dir

    # Logging
    wandb_project: str = 'moe-bf16-experiments_v2'
    wandb_run_name: Optional[str] = None

    # Optimizer
    optimizer: str = 'adamw'        # 'adamw' | 'muon_sophia' | 'sophia'
    muon_lr: float = 3e-2
    muon_wd: float = 1e-2
    sophia_lr: float = 1e-3
    sophia_b1: float = 0.965
    sophia_b2: float = 0.99
    sophia_rho: float = 0.1
    sophia_wd: float = 0.2
    sophia_k: int = 10              # Hessian EMA update interval (steps)


class HFStreamingBatcher:
    """Stream dataset, tokenize, and expose [x,y] batches for train/eval."""
    def __init__(self,
                 dataset_name: str,
                 dataset_config: str,
                 tokenizer,
                 block_size: int,
                 device: torch.device,
                 seed: int,
                 text_key: str = 'text',
                 shuffle_buffer: int = 8192,
                 eval_sequences: int = 512):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.device = device
        self.seed = seed
        self.text_key = text_key
        self.shuffle_buffer = shuffle_buffer
        self.tokens_per_window = self.block_size + 1
        self._train_iter = self._sequence_iter(seed_offset=0)
        needed = max(1, int(eval_sequences))
        self._eval_chunks = list(itertools.islice(self._sequence_iter(seed_offset=1), needed))
        if len(self._eval_chunks) < needed:
            raise RuntimeError(f"Unable to materialize {needed} eval sequences from {dataset_name}:{dataset_config}")
        self._eval_index = 0

    def _new_stream(self, seed_offset: int):
        ds = load_dataset(self.dataset_name, name=self.dataset_config, split='train', streaming=True)
        try:
            ds = ds.shuffle(seed=self.seed + seed_offset, buffer_size=self.shuffle_buffer)
        except Exception:
            pass
        return iter(ds)

    def _tokenize_example(self, example) -> List[int]:
        text = ''
        if isinstance(example, dict):
            text = example.get(self.text_key) or ''
        else:
            text = str(example)
        if not text:
            return []
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_id is not None:
            ids.append(eos_id)
        return ids

    def _sequence_iter(self, seed_offset: int):
        stream = self._new_stream(seed_offset)
        buffer: List[int] = []
        while True:
            try:
                example = next(stream)
            except StopIteration:
                stream = self._new_stream(seed_offset)
                continue
            ids = self._tokenize_example(example)
            if not ids:
                continue
            buffer.extend(ids)
            while len(buffer) >= self.tokens_per_window:
                chunk = torch.tensor(buffer[:self.tokens_per_window], dtype=torch.long)
                del buffer[:self.block_size]
                yield chunk

    def _next_train_chunk(self) -> torch.Tensor:
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = self._sequence_iter(seed_offset=0)
            return next(self._train_iter)

    def _next_eval_chunk(self) -> torch.Tensor:
        chunk = self._eval_chunks[self._eval_index]
        self._eval_index = (self._eval_index + 1) % len(self._eval_chunks)
        return chunk

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = [self._next_train_chunk() if split == 'train' else self._next_eval_chunk() for _ in range(batch_size)]
        x = torch.stack([c[:-1] for c in chunks]).to(self.device, non_blocking=True)
        y = torch.stack([c[1:] for c in chunks]).to(self.device, non_blocking=True)
        return x, y


def cosine_lr(it: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)
    if it >= total:
        return min_lr
    a = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * a))


def _anneal_linear(it: int, init_v: float, final_v: float, total: int) -> float:
    if total <= 0:
        return final_v
    if it >= total:
        return final_v
    a = it / float(total)
    return (1 - a) * init_v + a * final_v


def evaluate(model: TinyMoETransformer, data: HFStreamingBatcher, cfg: TrainConfig, eval_iters: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data.get_batch('val', cfg.batch_size)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def _save_ckpt(path: Path, model: TinyMoETransformer, optimizer: torch.optim.Optimizer, cfg: TrainConfig, it: int, val_loss: Optional[float] = None):
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'cfg': asdict(cfg),
        'iter': int(it),
    }
    if val_loss is not None:
        payload['val_loss'] = float(val_loss)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _maybe_resume(model: TinyMoETransformer, optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> Tuple[int, float]:
    start_iter = 0
    best_val = float('inf')
    ckpt: Optional[Path] = None
    if cfg.resume:
        p = Path(cfg.resume)
        ckpt = p if p.is_file() else (p / 'last_moe_bf16.pt')
    elif cfg.auto_resume:
        cand = Path(cfg.checkpoint_dir) / 'last_moe_bf16.pt'
        ckpt = cand if cand.exists() else None
    if ckpt and ckpt.exists():
        obj = torch.load(ckpt, map_location='cpu')
        try:
            model.load_state_dict(obj['model'], strict=False)
        except Exception:
            model.load_state_dict(obj['model'], strict=False)
        if 'optimizer' in obj:
            try:
                optimizer.load_state_dict(obj['optimizer'])
            except Exception:
                pass
        start_iter = int(obj.get('iter', 0))
        best_val = float(obj.get('val_loss', float('inf')))
        print(f"[resume] Loaded checkpoint from {ckpt} at iter={start_iter} best_val={best_val}")
    return start_iter, best_val


def _split_params_for_muon(model: torch.nn.Module):
    hidden, aux = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('wte' in name) or ('lm_head' in name):
            aux.append(p)
        elif p.ndim >= 2:
            hidden.append(p)
        else:
            aux.append(p)
    return hidden, aux


class _MuonSophia:
    """Muon on hidden params, SophiaG on aux params (single device)."""
    def __init__(self, hidden, aux, muon_kw, sophia_kw):
        from muon import SingleDeviceMuon
        from sophia import SophiaG
        self.muon = SingleDeviceMuon(hidden, **muon_kw)
        self.sophia = SophiaG(aux, **sophia_kw)
        # mark groups to allow per-optimizer LR scheduling
        for _g in self.muon.param_groups: _g['kind'] = 'muon'
        for _g in self.sophia.param_groups: _g['kind'] = 'sophia'
        # expose combined param groups so schedulers can update lr, wd, etc.
        self.param_groups = list(self.muon.param_groups) + list(self.sophia.param_groups)

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

    # checkpoint compatibility
    def state_dict(self):
        return {
            'muon': self.muon.state_dict(),
            'sophia': self.sophia.state_dict(),
        }

    def load_state_dict(self, state):
        try:
            if isinstance(state, dict) and 'muon' in state and 'sophia' in state:
                self.muon.load_state_dict(state['muon'])
                self.sophia.load_state_dict(state['sophia'])
            else:
                # allow passing straight-through if caller saved underlying optimizers separately
                if isinstance(state, dict) and 'state' in state:
                    # likely an AdamW-style state dict; ignore
                    return
        except Exception:
            pass


def _build_optimizer(cfg: TrainConfig, model: TinyMoETransformer):
    kind = (cfg.optimizer or 'adamw').lower()
    if kind in {'adamw', 'adam'}:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        sophia_meta = None
        return optim, sophia_meta
    if kind in {'muon_sophia', 'muon+sophia', 'ms'}:
        hidden, aux = _split_params_for_muon(model)
        opt = _MuonSophia(
            hidden, aux,
            muon_kw=dict(lr=cfg.muon_lr, weight_decay=cfg.muon_wd, momentum=0.95),
            sophia_kw=dict(lr=cfg.sophia_lr, betas=(cfg.sophia_b1, cfg.sophia_b2), rho=cfg.sophia_rho, weight_decay=cfg.sophia_wd),
        )
        sophia_meta = {'k_hess': int(cfg.sophia_k)}
        return opt, sophia_meta
    if kind in {'sophia', 'sophia_g', 'sophiag'}:
        from sophia import SophiaG
        opt = SophiaG(model.parameters(), lr=cfg.sophia_lr, betas=(cfg.sophia_b1, cfg.sophia_b2), rho=cfg.sophia_rho, weight_decay=cfg.sophia_wd)
        sophia_meta = {'k_hess': int(cfg.sophia_k)}
        return opt, sophia_meta
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train(cfg: TrainConfig):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Data
    eval_sequences = max(cfg.eval_take, cfg.eval_iters * cfg.batch_size)
    data = HFStreamingBatcher(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        tokenizer=tokenizer,
        block_size=cfg.block_size,
        device=device,
        seed=cfg.seed,
        text_key=cfg.text_key,
        shuffle_buffer=cfg.shuffle_buffer,
        eval_sequences=eval_sequences,
    )

    # Model
    cfg.vocab_size = len(tokenizer)
    model = TinyMoETransformer(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_heads=cfg.n_kv_heads,
        d_model=cfg.d_model,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        bias=cfg.bias,
        n_experts=cfg.n_experts,
        capacity_factor=cfg.capacity_factor,
        dropless=cfg.dropless,
        load_balance_alpha=cfg.load_balance_alpha,
        router_z_loss_coef=cfg.router_z_loss_coef,
        attn_gate=cfg.attn_gate,
        router_temperature=cfg.router_temp_init,
        router_noise_std=cfg.router_noise_std_init,
        router_noise_type=cfg.router_noise_type,
        use_rope=cfg.use_rope,
        rope_theta=cfg.rope_theta,
        moe_grouped=cfg.moe_grouped,
        qk_norm=cfg.qk_norm,
        qk_norm_eps=cfg.qk_norm_eps,
        compile_submodules=bool(cfg.compile),
        fp8=bool(cfg.fp8),
    ).to(device=device, dtype=torch.bfloat16)

    total_params = model.num_parameters()
    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Config: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, experts={cfg.n_experts}")

    # Configure inductor for dynamic shapes; warmup to trigger per-submodule compile
    if cfg.compile:
        try:
            import torch._inductor.config as inductor_config
            # Keep cudagraphs off initially for dynamic routing
            if hasattr(inductor_config.triton, 'cudagraphs'):
                inductor_config.triton.cudagraphs = False
            if hasattr(inductor_config, 'shape_padding'):
                inductor_config.shape_padding = True
            if hasattr(inductor_config.triton, 'cudagraph_skip_dynamic_graphs'):
                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            if hasattr(inductor_config.triton, 'cudagraph_dynamic_shape_warn_limit'):
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
        except Exception:
            pass
        # warmup pass to seed dynamic guards
        try:
            x_warm, y_warm = data.get_batch('train', cfg.batch_size)
            try:
                import torch._dynamo as dynamo
                dynamo.mark_dynamic(x_warm, 0)
                dynamo.mark_dynamic(x_warm, 1)
            except Exception:
                pass
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _ = model(x_warm, y_warm)
        except Exception as e:
            print(f"compile warmup failed: {e}")

    optimizer, sophia_meta = _build_optimizer(cfg, model)

    logger = WandBLogger(project=cfg.wandb_project, run_name=cfg.wandb_run_name)
    logger.watch(model)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_iter, best_val = _maybe_resume(model, optimizer, cfg)

    tokens_cum = 0
    t0 = time.time()
    clip_cum = 0

    for it in range(start_iter, cfg.max_iters + 1):
        # LR schedule
        lr = cosine_lr(it, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
        if hasattr(optimizer, 'param_groups'):
            if sophia_meta is not None:
                # derive per-optimizer min_lrs by keeping the same min/base ratio
                ratio = (cfg.min_lr / max(1e-12, cfg.learning_rate))
                muon_lr_sched = cosine_lr(it, cfg.muon_lr, cfg.muon_lr * ratio, cfg.warmup_iters, cfg.lr_decay_iters)
                sophia_lr_sched = cosine_lr(it, cfg.sophia_lr, cfg.sophia_lr * ratio, cfg.warmup_iters, cfg.lr_decay_iters)
                # if groups are annotated, set per kind; else assume sophia-only
                any_kind = any(isinstance(g, dict) and ('kind' in g) for g in optimizer.param_groups)
                for pg in optimizer.param_groups:
                    if any_kind:
                        if pg.get('kind') == 'muon':
                            pg['lr'] = muon_lr_sched
                        else:
                            pg['lr'] = sophia_lr_sched
                    else:
                        pg['lr'] = sophia_lr_sched
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

        # Router dynamics schedule per-step
        curr_temp = _anneal_linear(it, cfg.router_temp_init, cfg.router_temp_final, cfg.router_temp_anneal_iters)
        curr_noise = _anneal_linear(it, cfg.router_noise_std_init, 0.0, cfg.router_noise_decay_iters)
        for blk in model.h:
            if hasattr(blk, 'moe'):
                blk.moe.router.set_router_state(curr_temp, curr_noise)
                blk.moe.router.noise_type = cfg.router_noise_type

        if hasattr(optimizer, 'zero_grad'):
            optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_router_metrics = None
        me_list, sv_list = [], []
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = data.get_batch('train', cfg.batch_size)
            logits, loss = model(x, y)
            loss = loss / max(1, cfg.gradient_accumulation_steps)
            loss.backward()
            total_loss += float(loss.item())

            # Router stats from the last micro-step only
            with torch.no_grad():
                auxs, max_fracs, num_active, drops, tpm, ent = [], [], [], [], [], []
                for blk in model.h:
                    if hasattr(blk, 'moe') and hasattr(blk.moe, '_last_stats'):
                        st = blk.moe._last_stats
                        auxs.append(float(st['aux']))
                        max_fracs.append(float(st['max_frac']))
                        num_active.append(int(st['num_active']))
                        drops.append(float(st['drop_frac']))
                        tpm.append(float(st['top1_p_mean']))
                        ent.append(float(st['entropy_mean']))
                        if 'me' in st and 'served' in st:
                            me_list.append(st['me'].float().cpu())
                            sv_list.append(st['served'].float().cpu())
                if auxs:
                    last_router_metrics = {
                        'router/aux_mean': sum(auxs) / len(auxs),
                        'router/max_frac_mean': sum(max_fracs) / len(max_fracs),
                        'router/max_frac_max': max(max_fracs),
                        'router/active_min': min(num_active),
                        'router/active_mean': sum(num_active) / len(num_active),
                        'router/drop_frac_mean': sum(drops) / len(drops),
                        'router/top1_p_mean': sum(tpm) / len(tpm),
                        'router/entropy_mean': sum(ent) / len(ent),
                    }
                    last_router_metrics['router/collapsed'] = 1 if last_router_metrics['router/max_frac_max'] >= 0.90 or last_router_metrics['router/active_min'] <= 1 else 0

        # Gradient clip
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(norm):
                clip_cum += int(norm > cfg.grad_clip)

        # Optimizer step (Sophia variants need bs=tokens/iter)
        try:
            optimizer.step(bs=int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps))
        except TypeError:
            optimizer.step()

        # Sophia Hessian EMA update
        if sophia_meta is not None:
            k = max(1, int(sophia_meta.get('k_hess', 10)))
            if (it + 1) % k == 0:
                x_s, _ = data.get_batch('train', cfg.batch_size)
                model.zero_grad(set_to_none=True)
                logits, _ = model(x_s, None)
                with torch.no_grad():
                    y_samp = torch.distributions.Categorical(logits=logits.float()).sample()
                loss_s = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_samp.reshape(-1))
                loss_s.backward()
                try:
                    optimizer.update_hessian()
                except Exception:
                    pass

        # Logging every cfg.log_interval
        if (it % cfg.log_interval) == 0 and it > start_iter:
            step_tokens = int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps)
            tokens_cum += step_tokens
            tps = step_tokens * cfg.log_interval / max(1e-6, (time.time() - t0))
            avg_loss = total_loss
            metrics = {
                'train/loss': avg_loss,
                'lr': lr,
                'router/temp': curr_temp,
                'router/noise_std': curr_noise,
                'grad/clipped_cum': int(clip_cum),
                'train/tokens_b': tokens_cum / 1e9,
            }
            if sophia_meta is not None and hasattr(optimizer, 'param_groups'):
                try:
                    mu_lr = next((pg['lr'] for pg in optimizer.param_groups if pg.get('kind')=='muon'), None)
                    so_lr = next((pg['lr'] for pg in optimizer.param_groups if pg.get('kind')=='sophia'), None)
                    if mu_lr is not None: metrics['opt/lr_muon'] = float(mu_lr)
                    if so_lr is not None: metrics['opt/lr_sophia'] = float(so_lr)
                except Exception:
                    pass

            if last_router_metrics:
                metrics.update(last_router_metrics)

            # Aggregate and log per-expert demand vs served fractions
            if me_list:
                _torch = torch
                me_avg = _torch.stack(me_list).mean(0)
                sv_avg = _torch.stack(sv_list).mean(0)
                for e in range(len(me_avg)):
                    metrics[f'router/frac_dem_e{e}'] = float(me_avg[e])
                    metrics[f'router/frac_srv_e{e}'] = float(sv_avg[e])
                metrics['router/frac_dem_max'] = float(me_avg.max())
                metrics['router/frac_srv_max'] = float(sv_avg.max())
                metrics['router/active_dem'] = int((me_avg > 0).sum())
                metrics['router/active_srv'] = int((sv_avg > 0).sum())

                # Combined utilization plot (served vs demand across experts)
                try:
                    import wandb as _wb
                    xs = list(range(len(me_avg)))
                    ys = [[float(v) for v in sv_avg.tolist()], [float(v) for v in me_avg.tolist()]]
                    util_plot = _wb.plot.line_series(xs=xs, ys=ys, keys=['served','demand'], title='Expert Utilization', xname='expert')
                    metrics['router/util_plot'] = util_plot
                    metrics['router/frac_srv_sum'] = float(sv_avg.sum())
                    metrics['router/frac_dem_sum'] = float(me_avg.sum())
                except Exception:
                    pass

            logger.log_metrics(metrics, step=it)

            if last_router_metrics:
                rf = last_router_metrics['router/max_frac_max']
                na = last_router_metrics['router/active_min']
                col = last_router_metrics['router/collapsed']
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s | r_max {rf:.2f} act_min {na} col {col}")
            else:
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s")
            t0 = time.time()

        # Eval + checkpoint
        if (it % cfg.eval_interval) == 0 and it > start_iter:
            val_loss = evaluate(model, data, cfg, cfg.eval_iters)
            logger.log_metrics({'val/loss': val_loss, 'val/ppl': math.exp(min(20.0, val_loss))}, step=it)
            print(f"eval | val_loss {val_loss:.4f}")
            # save 'last'
            _save_ckpt(ckpt_dir / 'last_moe_bf16.pt', model, optimizer, cfg, it, val_loss)
            # save 'best' on improvement
            if val_loss < best_val:
                best_val = val_loss
                _save_ckpt(ckpt_dir / 'best_moe_bf16.pt', model, optimizer, cfg, it, val_loss)

        # Periodic checkpointing
        if cfg.save_interval and it > start_iter and (it % cfg.save_interval) == 0:
            tag = f"iter_{it:06d}.pt"
            _save_ckpt(ckpt_dir / tag, model, optimizer, cfg, it, None)
            if cfg.keep_checkpoints and cfg.keep_checkpoints > 0:
                ckpts = sorted(ckpt_dir.glob('iter_*.pt'))
                if len(ckpts) > cfg.keep_checkpoints:
                    for pth in ckpts[:len(ckpts) - cfg.keep_checkpoints]:
                        try:
                            pth.unlink()
                        except Exception:
                            pass

    logger.set_summary(best_val_loss=best_val, params=total_params)
    logger.finish()


def main():
    import argparse
    p = argparse.ArgumentParser(description='BF16 Tiny MoE Transformer trainer')
    # Model
    p.add_argument('--vocab_size', type=int, default=32768)
    p.add_argument('--n_layer', type=int, default=10)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_kv_heads', type=int, default=None)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_experts', type=int, default=4)
    p.add_argument('--block_size', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', action='store_true')
    p.add_argument('--capacity_factor', type=float, default=1.25)
    p.add_argument('--dropless', action='store_true')
    p.add_argument('--no-dropless', dest='dropless', action='store_false')
    p.set_defaults(dropless=True)
    p.add_argument('--load_balance_alpha', type=float, default=0.05)
    p.add_argument('--router_z_loss_coef', type=float, default=0.0)
    p.add_argument('--attn_gate', type=str, default='none', choices=['none', 'sigmoid_head'])
    p.add_argument('--use_rope', dest='use_rope', action='store_true')
    p.add_argument('--no-use_rope', dest='use_rope', action='store_false')
    p.set_defaults(use_rope=True)
    p.add_argument('--rope_theta', type=float, default=10000.0)
    # QK-Norm
    p.add_argument('--qk_norm', action='store_true')
    p.add_argument('--qk_norm_eps', type=float, default=1e-6)
    # Router dynamics CLI
    p.add_argument('--router_temp_init', type=float, default=1.5)
    p.add_argument('--router_temp_final', type=float, default=1.0)
    p.add_argument('--router_temp_anneal_iters', type=int, default=1000)
    p.add_argument('--router_noise_std_init', type=float, default=0.5)
    p.add_argument('--router_noise_decay_iters', type=int, default=1000)
    p.add_argument('--router_noise_type', type=str, default='gumbel', choices=['gumbel', 'gaussian'])
    p.add_argument('--moe_grouped', action='store_true')

    # Train
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--max_iters', type=int, default=2000)
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--warmup_iters', type=int, default=500)
    p.add_argument('--lr_decay_iters', type=int, default=2000)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--fp8', action='store_true')

    # Data
    p.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset_config', type=str, default='sample-10BT')
    p.add_argument('--text_key', type=str, default='text')
    p.add_argument('--shuffle_buffer', type=int, default=8192)
    p.add_argument('--eval_take', type=int, default=512)

    # IO
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe_bf16')
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_interval', type=int, default=0)
    p.add_argument('--keep_checkpoints', type=int, default=3)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--auto_resume', action='store_true')

    # Optimizer
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw','muon_sophia','sophia'])
    p.add_argument('--muon_lr', type=float, default=3e-2)
    p.add_argument('--muon_wd', type=float, default=1e-2)
    p.add_argument('--sophia_lr', type=float, default=1e-3)
    p.add_argument('--sophia_b1', type=float, default=0.965)
    p.add_argument('--sophia_b2', type=float, default=0.99)
    p.add_argument('--sophia_rho', type=float, default=0.1)
    p.add_argument('--sophia_wd', type=float, default=0.2)
    p.add_argument('--sophia_k', type=int, default=10)

    # Logging
    p.add_argument('--wandb_project', type=str, default='moe-bf16-experiments_v2')
    p.add_argument('--wandb_run_name', type=str, default=None)

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))

    d = cfg.d_model
    params_per_expert = 8 * (d ** 2)
    print(f"Estimated params/expert (SwiGLU): ~{params_per_expert/1e6:.2f}M for d={d}")
    train(cfg)


if __name__ == '__main__':
    main()
