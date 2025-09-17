#!/usr/bin/env python3
"""
Training script for TinyMoETransformer (BF16) split out from model_experimental.py.

Expert Choice routing is removed; routing is top-1 token-choice only.
"""

import itertools
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

try:
    from wandb_logger import WandBLogger
except Exception:
    # Fallback no-op logger
    class WandBLogger:
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

    # Data
    dataset_name: str = 'HuggingFaceFW/fineweb-edu'
    dataset_config: str = 'sample-10BT'
    text_key: str = 'text'
    shuffle_buffer: int = 8192
    eval_take: int = 512

    # IO
    checkpoint_dir: str = 'checkpoints_moe_bf16'
    log_interval: int = 10

    # Logging
    wandb_project: str = 'moe-bf16-experiments'
    wandb_run_name: Optional[str] = None


class HFStreamingBatcher:
    """Stream Hugging Face dataset, tokenize, and expose train/eval batches."""
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
        ds = load_dataset(self.dataset_name, name=self.dataset_config, split="train", streaming=True)
        try:
            ds = ds.shuffle(seed=self.seed + seed_offset, buffer_size=self.shuffle_buffer)
        except Exception:
            pass
        return iter(ds)

    def _tokenize_example(self, example):
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
        buffer = []
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
        if split == 'train':
            chunks = [self._next_train_chunk() for _ in range(batch_size)]
        else:
            chunks = [self._next_eval_chunk() for _ in range(batch_size)]
        x = torch.stack([c[:-1] for c in chunks]).to(self.device, non_blocking=True)
        y = torch.stack([c[1:] for c in chunks]).to(self.device, non_blocking=True)
        return x, y

def cosine_lr(it: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / max(1, warmup)
    if it >= total:
        return min_lr
    progress = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


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
            logits, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train(cfg: TrainConfig):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
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
    ).to(device=device, dtype=torch.bfloat16)

    total_params = model.num_parameters()
    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Config: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, experts={cfg.n_experts}")
    compiled_enabled = False
    if cfg.compile and hasattr(torch, 'compile'):
        # Configure inductor to skip cudagraphs for dynamic shapes (MoE token counts vary per step)
        try:
            import torch._inductor.config as inductor_config
            if hasattr(inductor_config.triton, 'cudagraph_skip_dynamic_graphs'):
                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            if hasattr(inductor_config.triton, 'cudagraph_dynamic_shape_warn_limit'):
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
            if hasattr(inductor_config, 'shape_padding'):
                inductor_config.shape_padding = True
        except Exception:
            pass
        try:
            model = torch.compile(model, mode='max-autotune')
            compiled_enabled = True
        except Exception as e:
            print(f"torch.compile failed ({e}); continuing without compile.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    # Logging and checkpoints
    logger = WandBLogger(project=cfg.wandb_project, run_name=cfg.wandb_run_name)
    logger.watch(model)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val = float('inf')
    tokens_seen = 0
    tokens_cum = 0  # cumulative tokens for plotting x-axis in billions
    t0 = time.time()
    clip_cum = 0

    for it in range(cfg.max_iters):
        # LR schedule
        lr = cosine_lr(it, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Update router schedules (temperature and noise) per-step
        curr_temp = _anneal_linear(it, cfg.router_temp_init, cfg.router_temp_final, cfg.router_temp_anneal_iters)
        curr_noise = _anneal_linear(it, cfg.router_noise_std_init, 0.0, cfg.router_noise_decay_iters)
        for blk in model.h:
            if hasattr(blk, 'moe'):
                blk.moe.router.set_router_state(curr_temp, curr_noise)
                blk.moe.router.noise_type = cfg.router_noise_type

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_router_metrics = None
        me_list, sv_list = [], []
        for micro in range(cfg.gradient_accumulation_steps):
            x, y = data.get_batch('train', cfg.batch_size)
            logits, loss = model(x, y)
            loss = loss / max(1, cfg.gradient_accumulation_steps)
            loss.backward()
            total_loss += loss.item()

            # Gather router/expert stats from this micro-step (use the last one for logging)
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
                        # per-expert routed vs served fractions
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
                else:
                    last_router_metrics = None

        # Gradient clip
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(norm):
                clip_cum += int(norm > cfg.grad_clip)

        optimizer.step()

        # Logging
        step_tokens = int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps)
        tokens_seen += step_tokens
        tokens_cum += step_tokens
        if (it % cfg.log_interval) == 0:
            dt = max(1e-9, time.time() - t0)
            tps = tokens_seen / dt
            avg_loss = total_loss
            metrics = {
                'train/loss': avg_loss,
                'lr': lr,
                'router/temp': curr_temp,
                'router/noise_std': curr_noise,
            }
            metrics['grad/clipped_cum'] = int(clip_cum)
            # Aggregate and log per-expert demand vs served fractions
            if me_list:
                import torch as _torch
                me_avg = _torch.stack(me_list).mean(0)
                sv_avg = _torch.stack(sv_list).mean(0)
                for e in range(len(me_avg)):
                    metrics[f'router/frac_dem_e{e}'] = float(me_avg[e])
                    metrics[f'router/frac_srv_e{e}'] = float(sv_avg[e])
                metrics['router/frac_dem_max'] = float(me_avg.max())
                metrics['router/frac_srv_max'] = float(sv_avg.max())
                metrics['router/active_dem'] = int((me_avg > 0).sum())
                metrics['router/active_srv'] = int((sv_avg > 0).sum())
            # cumulative tokens in billions for x-axis
            metrics['train/tokens_b'] = tokens_cum / 1e9
            if last_router_metrics:
                metrics.update(last_router_metrics)
            logger.log_metrics(metrics, step=it)
            if last_router_metrics:
                rf = last_router_metrics['router/max_frac_max']
                na = last_router_metrics['router/active_min']
                col = last_router_metrics['router/collapsed']
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s | r_max {rf:.2f} act_min {na} col {col}")
            else:
                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s")
            t0 = time.time(); tokens_seen = 0

        # Eval
        if (it % cfg.eval_interval) == 0:
            val_loss = evaluate(model, data, cfg, cfg.eval_iters)
            logger.log_metrics({'val/loss': val_loss, 'val/ppl': math.exp(min(20.0, val_loss))}, step=it)
            print(f"eval | val_loss {val_loss:.4f}")
            try:
                torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'val_loss': float(val_loss), 'iter': it}, Path(cfg.checkpoint_dir) / 'last_moe_bf16.pt')
            except Exception:
                pass

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = Path(cfg.checkpoint_dir) / 'best_moe_bf16.pt'
                torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'val_loss': val_loss, 'iter': it}, ckpt_path)

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
    p.add_argument('--attn_gate', type=str, default='none', choices=['none', 'sigmoid_head'], help='Enable SDPA + elementwise head-specific sigmoid gate')
    p.add_argument('--use_rope', dest='use_rope', action='store_true')
    p.add_argument('--no-use_rope', dest='use_rope', action='store_false')
    p.set_defaults(use_rope=True)
    p.add_argument('--rope_theta', type=float, default=10000.0)
    # QK-Norm
    p.add_argument('--qk_norm', action='store_true', help='Apply RMSNorm to Q and K before RoPE')
    p.add_argument('--qk_norm_eps', type=float, default=1e-6)
    # Router dynamics CLI
    p.add_argument('--router_temp_init', type=float, default=1.5)
    p.add_argument('--router_temp_final', type=float, default=1.0)
    p.add_argument('--router_temp_anneal_iters', type=int, default=1000)
    p.add_argument('--router_noise_std_init', type=float, default=0.5)
    p.add_argument('--router_noise_decay_iters', type=int, default=1000)
    p.add_argument('--router_noise_type', type=str, default='gumbel', choices=['gumbel', 'gaussian'])
    p.add_argument('--moe_grouped', action='store_true', help='Use grouped/padded MoE with batched GEMMs (capacity mode)')

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

    # IO
    p.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset_config', type=str, default='sample-10BT')
    p.add_argument('--text_key', type=str, default='text')
    p.add_argument('--shuffle_buffer', type=int, default=8192)
    p.add_argument('--eval_take', type=int, default=512, help='Number of streaming sequences cached for eval (>= eval_iters * batch_size).')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe_bf16')
    p.add_argument('--log_interval', type=int, default=10)

    # Logging
    p.add_argument('--wandb_project', type=str, default='moe-bf16-experiments')
    p.add_argument('--wandb_run_name', type=str, default=None)

    args = p.parse_args()
    cfg = TrainConfig(**vars(args))

    # Print a quick expert size estimate
    d = cfg.d_model
    params_per_expert = 8 * (d ** 2)
    print(f"Estimated params/expert (SwiGLU): ~{params_per_expert/1e6:.2f}M for d={d}")
    train(cfg)


if __name__ == '__main__':
    main()
