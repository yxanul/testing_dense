**MoE Router Metrics Guide**

This guide explains the key router/expert metrics logged during training and provides practical target ranges and tuning tips. Assumptions:

- Top‑1 routing (token chooses one expert).
- Default is dropless routing; capacity mode may be enabled in some runs.
- Number of experts = E. Uniform/ideal balance = each expert handles ~1/E of tokens.

**Key Concepts**
- Uniform baseline: if routing is perfectly balanced and not too peaky, demand/served per expert ≈ 1/E.
- Collapse: one/few experts receive most tokens while others are idle. In code, we flag collapse when `router/max_frac_max ≥ 0.90` or `router/active_min ≤ 1`.
- Peaked vs. diffuse: controlled by router temperature/noise. Too peaked can drive collapse; too diffuse reduces specialization.

**Metric Reference**
- `router/top1_p_mean`
  - Meaning: Mean probability assigned to the selected expert (softmax top‑1) across tokens.
  - Uniform baseline: ≈ 1/E (e.g., E=8 → 0.125).
  - Healthy range: ~1.5/E up to ~0.6. Example E=8 → ~0.18–0.6.
  - Watchouts: >0.8 sustained → router too confident/peaky; risk of collapse. <~1.2/E → too diffuse.
  - Levers: increase `router_temp_init/final` or `router_noise_std_init` to reduce peaking; decrease to sharpen if too diffuse.

- `router/max_frac_mean` and `router/max_frac_max`
  - Meaning: For each block, fraction of tokens sent to the most‑used expert; we log mean and max across blocks.
  - Uniform baseline: ≈ 1/E.
  - Healthy range: ≤ ~2/E most of training with occasional spikes. Example E=8 → ≤ ~0.25 typical, spikes <~0.35.
  - Watchouts: Sustained >~0.35–0.4 indicates imbalance; >~0.5 is severe; `≥ 0.90` is hard collapse (code flag).
  - Levers: raise `load_balance_alpha` (e.g., 0.1–0.2), add/raise `router_z_loss_coef` (e.g., 1e‑3→5e‑3), increase router temperature/noise.

- `router/frac_srv_max` and `router/frac_dem_max`
  - Meaning: Max served and max demanded expert fractions (averaged across blocks) at a log step.
  - Healthy: Both close to each other and near 1/E. In dropless they should match; in capacity mode served may trail demand.
  - Watchouts: Large persistent `frac_dem_max - frac_srv_max` → capacity too tight; increase `capacity_factor` or enable dropless.

- `router/entropy_mean`
  - Meaning: Mean entropy (nats) of router softmax over experts. Range: 0 to ln(E). Example E=8 → 0..~2.079.
  - Healthy: Mid‑range—neither near 0 (over‑confident) nor near ln(E) (uniform). Roughly ~0.6·ln(E) down to ~0.2–0.5·ln(E) as specialization grows.
  - Watchouts: ≪0.2·ln(E) early and falling fast → risk of peaky/collapse; ≈ln(E) late in training → under‑specialization.
  - Levers: adjust temperature/noise; balancing losses (`load_balance_alpha`, `router_z_loss_coef`).

- `router/aux_mean`
  - Meaning: Average load‑balancing auxiliary loss (already multiplied by `load_balance_alpha`) across blocks.
  - Healthy: Small and generally trending downward; occasional bumps when distribution shifts.
  - Watchouts: Growing while `max_frac_*` worsens → balancing too weak; increase `load_balance_alpha` or add `router_z_loss_coef`.

- `router/active_min` (printed) and per‑expert activity (`router/frac_dem_e*`, `router/frac_srv_e*`)
  - Meaning: Minimum number of experts that received any tokens across blocks, and per‑expert fractions.
  - Healthy: `active_min` close to E (E or E‑1) most steps; per‑expert lines overlap near 1/E with manageable variance.
  - Watchouts: Frequent dips of `active_min` to 1–2 → collapse onset.

- `router/drop_frac_mean`
  - Meaning: Fraction of routed tokens dropped due to capacity (capacity mode only).
  - Healthy: ≈0 in dropless; <~1–2% in capacity mode. If larger, raise `capacity_factor` or reduce peaking.

- `grad/clipped_cum`
  - Meaning: Cumulative count of steps where gradient norm exceeded `grad_clip`.
  - Healthy: Low incidence. Rule of thumb: <~5–10% of steps clipping over long runs.
  - Watchouts: Frequent clipping → lower LR, increase warmup, or reduce `grad_clip` cautiously; verify loss spikes/NaNs.

**Quick Thresholds (E=8)**
- Uniform target per‑expert ≈ 0.125.
- `top1_p_mean`: ~0.18–0.6 healthy; >0.8 risky.
- `max_frac_mean`: ≤0.25 typical; sustained >0.35 investigate.
- `max_frac_max`: spikes >0.4 okay briefly; sustained >0.5 bad; ≥0.9 collapsed.
- `frac_srv_max` vs `frac_dem_max`: should be close; diverging → capacity too tight.
- `entropy_mean`: ~0.4–1.6 nats (context‑dependent); avoid ~0 and ~2.08 extremes for long.
- `drop_frac_mean`: ~0 (dropless); <~0.02 (capacity).
- `grad/clipped_cum`: clipping ratio <~0.05–0.10 over time.

**Tuning Cheatsheet**
- Reduce peaking / avoid collapse:
  - Increase `router_temp_init/final` (e.g., +0.2–0.5).
  - Increase `router_noise_std_init` or keep noise high longer.
  - Increase `load_balance_alpha` (e.g., 0.1–0.2) and/or `router_z_loss_coef` (1e‑3→5e‑3).
- Improve balance without over‑diffusing:
  - Slightly raise temperature/noise and monitor `top1_p_mean` and `entropy_mean`.
  - Use dropless or increase `capacity_factor` if drops rise.
- Too diffuse / low specialization:
  - Decrease temperature/noise moderately; ensure `aux_mean` stays small and `max_frac_*` doesn’t spike.

**Reading the Dashboard**
- Overlay `router/frac_srv_e*` (and `router/frac_dem_e*`) in a single line plot to visualize utilization balance.
- Watch the built‑in `router/collapsed` flag; any non‑zero period warrants a quick parameter check.

These are pragmatic ranges, not hard rules-use them as guardrails and tune based on loss curves, throughput, and validation metrics.

**Experiment 1**
- Setup: `python train_experimental.py --device cuda --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --n_experts 8 --dropless --load_balance_alpha 0.12 --router_z_loss_coef 1e-3 --router_temp_init 2.2 --router_temp_final 1.3 --router_temp_anneal_iters 5000 --router_noise_std_init 0.9 --router_noise_decay_iters 5000 --router_noise_type gumbel --attn_gate sigmoid_head --qk_norm --use_rope --batch_size 8 --gradient_accumulation_steps 8 --learning_rate 8e-4 --max_iters 8000 --lr_decay_iters 8000 --warmup_iters 1000 --eval_interval 200 --eval_iters 50 --wandb_project moe-bf16-experiments_v2`
- Snapshot: observations at approximately step 400 (estimated from the dashboard screenshot; use W&B table export for exact values).

**Step ~400 Observations**
- train/loss: ~5.6–5.8 and decreasing.
- router/top1_p_mean: ~0.16–0.165 (uniform baseline at E=8 is 0.125).
- router/max_frac_mean: ~0.17; earlier spike near ~0.19 around step ~120, then stabilized.
- router/entropy_mean: ~1.97–1.99 nats (ln(8)=2.079), trending down gradually.
- router/aux_mean: ~0.125–0.126, with prior transient up to ~0.131.
- router/frac_srv_e*: per-expert served fractions clustered ~0.115–0.14; no persistent outlier.
- grad/clipped_cum: not shown in snapshot; keep clipping incidence under ~5–10% of steps.

**Assessment**
- Balance: Healthy, non-collapsed routing; experts utilized close to 1/E with modest variance.
- Specialization: Moderate (top1_p_mean above 1/E but far below peaky levels); entropy still high early, declining as annealing proceeds.
- Regularization: aux loss small and stable; schedules appear well tuned for this phase.

**Recommendations**
- Continue current anneal/noise schedules. Monitor:
  - router/max_frac_max (sustained >~0.4 warrants action) and active_min dips.
  - In capacity mode (if enabled later), watch `frac_dem_max - frac_srv_max` and `drop_frac_mean` (<~0.02 preferred).
- If specialization remains too diffuse beyond ~1k steps (entropy ~≥2.0), consider lowering `router_temp_final` to ~1.2–1.25.
- If imbalance increases, raise `load_balance_alpha` to 0.15–0.2 or keep noise higher for longer.

**Gradient Clipping (Experiment 1)**
- Observed `grad/clipped_cum` ≈ 320 by step ≈ 480 → clip ratio ≈ 65–70% of update steps.
- Guideline: prefer <~5–10% over long runs (short spikes are fine). Current level is high.
- Options to reduce clipping:
  - Lower LR to `6e-4` (or widen warmup to 1500–2000 steps) while keeping decay horizon matched to total iters.
  - Increase `grad_clip` cautiously to `1.5–2.0` and monitor stability (loss spikes/NaNs).
  - If early‑phase only, extend warmup and recheck after step ~1200.

**Experiment 1 — Update near 2k steps**
- Snapshot: step range ~1.7k–1.9k based on the latest dashboard.

- train/loss
  - Trend: continues to decrease from ~10+ to ~4.0 by ~1.7k steps.
  - Read: optimization is healthy; no signs of instability.

- router/max_frac_mean
  - Trend: early spike near ~0.19 around ~100–150, then gradual decline to ~0.155–0.16 at ~1.7k.
  - Read: balance improving over time; no evidence of collapse.

- router/top1_p_mean
  - Trend: rises to ~0.175 around ~500–700, then declines to ~0.152 by ~1.7k.
  - Read: specialization peaked early and relaxed slightly; still above 1/E (0.125), which is fine.

- router/entropy_mean
  - Trend: decreases from ~2.06 to a low near ~1.94 around ~800, then drifts up again toward ~2.04–2.06 by ~1.7k.
  - Read: distribution became sharper mid‑run then diffused somewhat; likely driven by balancing/temperature/noise schedules.

- router/aux_mean
  - Trend: brief early spike (~0.131), then steady decay to ~0.123–0.124 with small noise.
  - Read: balancing pressure is present but not dominating.

- router/frac_srv_e*
  - Trend: per‑expert served fractions remain clustered ~0.115–0.135 without a persistent outlier.
  - Read: healthy utilization spread across experts.

- router/max_frac_max, router/frac_dem_max, router/frac_srv_max
  - Not shown in the snapshot; verify they track the same improvement (expect `max_frac_max` well below collapse thresholds and `frac_srv_max ≈ frac_dem_max` in dropless mode).

- grad/clipped_cum
  - Re‑check at ~2k steps. If the slope has flattened compared to the early phase, the LR/warmup changes may be optional. If still steep, prefer LR/warmup adjustment over raising the clip.

Takeaways and next steps
- Non‑collapse and good balance are maintained; loss improves steadily.
- If you want stronger specialization past ~1.5–2k steps (entropy drifting back up):
  - Lower `router_temp_final` slightly (e.g., 1.25) or speed up the temperature anneal tail.
  - Reduce balancing slightly after ~2k (e.g., `load_balance_alpha` → 0.10 then 0.08) or lower `router_z_loss_coef` to `5e-4`, watching `max_frac_*` for regressions.
- If balance begins to worsen (sustained `max_frac_max > 0.4`), undo the above and bias toward more balancing or higher temperature/noise.

**Baseline (2k‑step reference)**
- Status: Healthy. Clipping plateaued from ~1k→2k; `router/max_frac_max` stayed well below collapse thresholds; per‑expert served fractions remain near 1/E.
- Milestone target: `val_loss < 4.0` by iteration ~1400 (observed: `eval | val_loss 3.9577`). Use this as a gating criterion for future A/B runs.
- Recommended seed: 1337 (already defaulted across NumPy, Torch, and dataset shuffle in the streaming loader). Pass explicitly for reproducibility: `--seed 1337`.
- Reproduce baseline command:
  - `python train_experimental.py --device cuda --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --n_experts 8 --dropless --load_balance_alpha 0.12 --router_z_loss_coef 1e-3 --router_temp_init 2.2 --router_temp_final 1.3 --router_temp_anneal_iters 5000 --router_noise_std_init 0.9 --router_noise_decay_iters 5000 --router_noise_type gumbel --attn_gate sigmoid_head --qk_norm --use_rope --batch_size 8 --gradient_accumulation_steps 8 --learning_rate 8e-4 --max_iters 8000 --lr_decay_iters 8000 --warmup_iters 1000 --eval_interval 200 --eval_iters 50 --wandb_project moe-bf16-experiments_v2 --seed 1337`
- Quick acceptance checklist (by ~1.4k steps):
  - `val/loss < 4.0`.
  - `router/max_frac_max` typically <~0.25 with no sustained upward trend; `router/max_frac_mean` ~0.155–0.165.
  - `router/top1_p_mean` ~0.15 ± 0.01; `router/entropy_mean` within ~1.94–2.06.
  - `router/frac_srv_e*` clustered ~0.115–0.135 without a persistent outlier.
  - `grad/clipped_cum` slope flat or slowly increasing after ~1k steps.



iter    180 | loss 7.0931 | lr 1.448e-04 | 33055 tok/s | r_max 0.20 act_min 8 col 0
iter    190 | loss 6.9327 | lr 1.528e-04 | 37180 tok/s | r_max 0.20 act_min 8 col 0
iter    200 | loss 6.9412 | lr 1.608e-04 | 33292 tok/s | r_max 0.21 act_min 8 col 0
eval | val_loss 6.8887
iter    210 | loss 6.7293 | lr 1.688e-04 | 29633 tok/s | r_max 0.21 act_min 8 col 0
iter    390 | loss 5.9859 | lr 3.128e-04 | 37070 tok/s | r_max 0.20 act_min 8 col 0
iter    400 | loss 5.8884 | lr 3.208e-04 | 33016 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 5.9093
iter    410 | loss 5.8852 | lr 3.288e-04 | 29806 tok/s | r_max 0.19 act_min 8 col 0
iter    420 | loss 5.7326 | lr 3.368e-04 | 37061 tok/s | r_max 0.21 act_min 8 col 0
iter    590 | loss 5.3854 | lr 4.728e-04 | 37105 tok/s | r_max 0.22 act_min 8 col 0
iter    600 | loss 5.2116 | lr 4.808e-04 | 33281 tok/s | r_max 0.21 act_min 8 col 0
eval | val_loss 5.2573
iter    610 | loss 5.3229 | lr 4.888e-04 | 29738 tok/s | r_max 0.21 act_min 8 col 0
iter    620 | loss 5.1967 | lr 4.968e-04 | 36982 tok/s | r_max 0.19 act_min 8 col 0
iter    790 | loss 4.8846 | lr 6.328e-04 | 36902 tok/s | r_max 0.19 act_min 8 col 0
iter    800 | loss 4.8271 | lr 6.408e-04 | 33385 tok/s | r_max 0.20 act_min 8 col 0
eval | val_loss 4.6967
iter    810 | loss 4.8088 | lr 6.488e-04 | 29638 tok/s | r_max 0.19 act_min 8 col 0
iter    820 | loss 4.8445 | lr 6.568e-04 | 37054 tok/s | r_max 0.20 act_min 8 col 0
iter    990 | loss 4.4986 | lr 7.928e-04 | 37044 tok/s | r_max 0.20 act_min 8 col 0
iter   1000 | loss 4.4125 | lr 8.000e-04 | 33066 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 4.3180
iter   1010 | loss 4.5170 | lr 8.000e-04 | 29817 tok/s | r_max 0.18 act_min 8 col 0
iter   1020 | loss 4.2907 | lr 8.000e-04 | 37181 tok/s | r_max 0.19 act_min 8 col 0
iter   1190 | loss 4.2914 | lr 7.986e-04 | 36883 tok/s | r_max 0.17 act_min 8 col 0
iter   1200 | loss 4.2599 | lr 7.985e-04 | 32979 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 4.1249
iter   1210 | loss 4.2722 | lr 7.983e-04 | 29761 tok/s | r_max 0.19 act_min 8 col 0
iter   1220 | loss 4.2198 | lr 7.981e-04 | 37069 tok/s | r_max 0.18 act_min 8 col 0
iter   1390 | loss 4.1034 | lr 7.941e-04 | 36614 tok/s | r_max 0.20 act_min 8 col 0
iter   1400 | loss 4.0014 | lr 7.938e-04 | 33460 tok/s | r_max 0.17 act_min 8 col 0
eval | val_loss 3.9577
iter   1410 | loss 4.2815 | lr 7.935e-04 | 29853 tok/s | r_max 0.19 act_min 8 col 0
iter   1420 | loss 4.0314 | lr 7.932e-04 | 37170 tok/s | r_max 0.18 act_min 8 col 0
iter   1590 | loss 3.9766 | lr 7.866e-04 | 37108 tok/s | r_max 0.19 act_min 8 col 0
iter   1600 | loss 3.9982 | lr 7.861e-04 | 32735 tok/s | r_max 0.18 act_min 8 col 0
eval | val_loss 3.8428
iter   1610 | loss 3.9787 | lr 7.857e-04 | 29795 tok/s | r_max 0.18 act_min 8 col 0
iter   1620 | loss 3.9973 | lr 7.852e-04 | 37033 tok/s | r_max 0.20 act_min 8 col 0
iter   1790 | loss 3.8391 | lr 7.761e-04 | 37129 tok/s | r_max 0.19 act_min 8 col 0
iter   1800 | loss 3.8393 | lr 7.755e-04 | 33279 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 3.7612
iter   1810 | loss 3.6285 | lr 7.748e-04 | 29907 tok/s | r_max 0.18 act_min 8 col 0
iter   1820 | loss 3.9918 | lr 7.742e-04 | 37015 tok/s | r_max 0.19 act_min 8 col 0
iter   1990 | loss 3.7622 | lr 7.626e-04 | 37210 tok/s | r_max 0.19 act_min 8 col 0
iter   2000 | loss 3.8298 | lr 7.619e-04 | 33155 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 3.6999
iter   2010 | loss 3.7276 | lr 7.611e-04 | 29629 tok/s | r_max 0.19 act_min 8 col 0
iter   2020 | loss 3.8116 | lr 7.604e-04 | 37116 tok/s | r_max 0.18 act_min 8 col 0
iter   2190 | loss 3.9544 | lr 7.464e-04 | 37092 tok/s | r_max 0.19 act_min 8 col 0
iter   2200 | loss 3.6836 | lr 7.455e-04 | 33077 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 3.6509
iter   2210 | loss 3.7907 | lr 7.446e-04 | 29898 tok/s | r_max 0.19 act_min 8 col 0
iter   2220 | loss 3.6495 | lr 7.437e-04 | 37110 tok/s | r_max 0.19 act_min 8 col 0


#!/usr/bin/env python3        try:            optimizer.step(bs=int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps))        except TypeError:            optimizer.step()        # Sophia Hessian EMA update        if sophia_meta is not None:            k = max(1, int(sophia_meta.get("k_hess", 10)))            if (it + 1) % k == 0:                x_s, _ = data.get_batch("train", cfg.batch_size)                model.zero_grad(set_to_none=True)                logits, _ = model(x_s, None)                with torch.no_grad():                    y_samp = torch.distributions.Categorical(logits=logits.float()).sample()                loss_s = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_samp.reshape(-1))                loss_s.backward()                try:                    optimizer.update_hessian()                except Exception:                    pass#!/usr/bin/env python3"""Training script for TinyMoETransformer (BF16) with HF streaming data and MoE router metrics.Adds robust checkpointing:- Save 'last' every eval- Save 'best' when val improves- Optional periodic saves via --save_interval (e.g., 500,1000,...)- Resume from checkpoint (--resume or --auto_resume)"""import mathimport timeimport itertoolsfrom dataclasses import dataclass, asdictfrom pathlib import Pathfrom typing import Optional, Tuple, Listimport numpy as npimport torchimport torch.nn.functional as Ffrom datasets import load_datasettry:    from wandb_logger import WandBLoggerexcept Exception:  # pragma: no cover    class WandBLogger:  # type: ignore        def __init__(self, *args, **kwargs): pass        def watch(self, *args, **kwargs): pass        def log_metrics(self, *args, **kwargs): pass        def log_eval(self, *args, **kwargs): pass        def set_summary(self, *args, **kwargs): pass        def finish(self, *args, **kwargs): passfrom model_experimental import TinyMoETransformer, tokenizer# --------------------------- Data + training ---------------------------@dataclassclass TrainConfig:    # Model    vocab_size: int = 32768    n_layer: int = 10    n_head: int = 8    n_kv_heads: Optional[int] = None    d_model: int = 512    n_experts: int = 4    block_size: int = 2048    dropout: float = 0.0    bias: bool = False    capacity_factor: float = 1.25    dropless: bool = True    load_balance_alpha: float = 0.05    router_z_loss_coef: float = 0.0    attn_gate: str = 'none'  # 'none' or 'sigmoid_head'    use_rope: bool = True    rope_theta: float = 10000.0    # QK-Norm    qk_norm: bool = False    qk_norm_eps: float = 1e-6    # Router dynamics    router_temp_init: float = 1.5    router_temp_final: float = 1.0    router_temp_anneal_iters: int = 1000    router_noise_std_init: float = 0.5    router_noise_decay_iters: int = 1000    router_noise_type: str = 'gumbel'  # 'gumbel' or 'gaussian'    moe_grouped: bool = False    # Training    device: str = 'cuda'    batch_size: int = 8    gradient_accumulation_steps: int = 1    max_iters: int = 2000    eval_interval: int = 200    eval_iters: int = 50    learning_rate: float = 3e-4    min_lr: float = 3e-5    weight_decay: float = 0.1    beta1: float = 0.9    beta2: float = 0.95    warmup_iters: int = 200    lr_decay_iters: int = 2000    grad_clip: float = 1.0    compile: bool = False    seed: int = 1337    # Data    dataset_name: str = 'HuggingFaceFW/fineweb-edu'    dataset_config: str = 'sample-10BT'    text_key: str = 'text'    shuffle_buffer: int = 8192    eval_take: int = 512    # IO    checkpoint_dir: str = 'checkpoints_moe_bf16'    log_interval: int = 10    save_interval: int = 0          # set >0 to save periodic checkpoints    keep_checkpoints: int = 3       # max periodic checkpoints to retain (0=keep all)    resume: Optional[str] = None    # path to checkpoint file or dir    auto_resume: bool = False       # auto-resume from last in checkpoint_dir    # Logging    wandb_project: str = 'moe-bf16-experiments_v2'    wandb_run_name: Optional[str] = None    # Optimizer    optimizer: str = 'adamw'        # 'adamw' or 'muon_sophia' or 'sophia'    # Muon params    muon_lr: float = 3e-2    muon_wd: float = 1e-2    # Sophia params    sophia_lr: float = 1e-3    sophia_b1: float = 0.965    sophia_b2: float = 0.99    sophia_rho: float = 0.1    sophia_wd: float = 0.2    sophia_k: int = 10              # Hessian EMA update interval (steps)class HFStreamingBatcher:    """Stream dataset, tokenize, and expose [x,y] batches for train/eval."""    def __init__(self,                 dataset_name: str,                 dataset_config: str,                 tokenizer,                 block_size: int,                 device: torch.device,                 seed: int,                 text_key: str = 'text',                 shuffle_buffer: int = 8192,                 eval_sequences: int = 512):        self.dataset_name = dataset_name        self.dataset_config = dataset_config        self.tokenizer = tokenizer        self.block_size = int(block_size)        self.device = device        self.seed = seed        self.text_key = text_key        self.shuffle_buffer = shuffle_buffer        self.tokens_per_window = self.block_size + 1        self._train_iter = self._sequence_iter(seed_offset=0)        needed = max(1, int(eval_sequences))        self._eval_chunks = list(itertools.islice(self._sequence_iter(seed_offset=1), needed))        if len(self._eval_chunks) < needed:            raise RuntimeError(f"Unable to materialize {needed} eval sequences from {dataset_name}:{dataset_config}")        self._eval_index = 0    def _new_stream(self, seed_offset: int):        ds = load_dataset(self.dataset_name, name=self.dataset_config, split='train', streaming=True)        try:            ds = ds.shuffle(seed=self.seed + seed_offset, buffer_size=self.shuffle_buffer)        except Exception:            pass        return iter(ds)    def _tokenize_example(self, example) -> List[int]:        text = ''        if isinstance(example, dict):            text = example.get(self.text_key) or ''        else:            text = str(example)        if not text:            return []        ids = self.tokenizer.encode(text, add_special_tokens=False)        eos_id = getattr(self.tokenizer, 'eos_token_id', None)        if eos_id is not None:            ids.append(eos_id)        return ids    def _sequence_iter(self, seed_offset: int):        stream = self._new_stream(seed_offset)        buffer: List[int] = []        while True:            try:                example = next(stream)            except StopIteration:                stream = self._new_stream(seed_offset)                continue            ids = self._tokenize_example(example)            if not ids:                continue            buffer.extend(ids)            while len(buffer) >= self.tokens_per_window:                chunk = torch.tensor(buffer[:self.tokens_per_window], dtype=torch.long)                del buffer[:self.block_size]                yield chunk    def _next_train_chunk(self) -> torch.Tensor:        try:            return next(self._train_iter)        except StopIteration:            self._train_iter = self._sequence_iter(seed_offset=0)            return next(self._train_iter)    def _next_eval_chunk(self) -> torch.Tensor:        chunk = self._eval_chunks[self._eval_index]        self._eval_index = (self._eval_index + 1) % len(self._eval_chunks)        return chunk    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:        chunks = [self._next_train_chunk() if split == 'train' else self._next_eval_chunk() for _ in range(batch_size)]        x = torch.stack([c[:-1] for c in chunks]).to(self.device, non_blocking=True)        y = torch.stack([c[1:] for c in chunks]).to(self.device, non_blocking=True)        return x, ydef cosine_lr(it: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:    if it < warmup:        return base_lr * (it + 1) / max(1, warmup)    if it >= total:        return min_lr    a = (it - warmup) / max(1, total - warmup)    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * a))def _anneal_linear(it: int, init_v: float, final_v: float, total: int) -> float:    if total <= 0:        return final_v    if it >= total:        return final_v    a = it / float(total)    return (1 - a) * init_v + a * final_vdef evaluate(model: TinyMoETransformer, data: HFStreamingBatcher, cfg: TrainConfig, eval_iters: int) -> float:    model.eval()    losses = []    with torch.no_grad():        for _ in range(eval_iters):            x, y = data.get_batch('val', cfg.batch_size)            _, loss = model(x, y)            losses.append(float(loss.item()))    model.train()    return float(sum(losses) / max(1, len(losses)))def _save_ckpt(path: Path, model: TinyMoETransformer, optimizer: torch.optim.Optimizer, cfg: TrainConfig, it: int, val_loss: Optional[float] = None):    payload = {        'model': model.state_dict(),        'optimizer': optimizer.state_dict(),        'cfg': asdict(cfg),        'iter': int(it),    }    if val_loss is not None:        payload['val_loss'] = float(val_loss)    path.parent.mkdir(parents=True, exist_ok=True)    torch.save(payload, path)def _maybe_resume(model: TinyMoETransformer, optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> Tuple[int, float]:    start_iter = 0    best_val = float('inf')    ckpt: Optional[Path] = None    if cfg.resume:        p = Path(cfg.resume)        ckpt = p if p.is_file() else (p / 'last_moe_bf16.pt')    elif cfg.auto_resume:        cand = Path(cfg.checkpoint_dir) / 'last_moe_bf16.pt'        ckpt = cand if cand.exists() else None    if ckpt and ckpt.exists():        obj = torch.load(ckpt, map_location='cpu')        try:            model.load_state_dict(obj['model'], strict=False)        except Exception:            model.load_state_dict(obj['model'], strict=False)        if 'optimizer' in obj:            try:                optimizer.load_state_dict(obj['optimizer'])            except Exception:                pass        start_iter = int(obj.get('iter', 0))        best_val = float(obj.get('val_loss', float('inf')))        print(f"[resume] Loaded checkpoint from {ckpt} at iter={start_iter} best_val={best_val}")    return start_iter, best_valdef _split_params_for_muon(model: torch.nn.Module):    hidden, aux = [], []    for name, p in model.named_parameters():        if not p.requires_grad:            continue        if ('wte' in name) or ('lm_head' in name):            aux.append(p)        elif p.ndim >= 2:            hidden.append(p)        else:            aux.append(p)    return hidden, auxclass _MuonSophia:    """Muon on hidden params, SophiaG on aux params (single device)."""    def __init__(self, hidden, aux, muon_kw, sophia_kw):        from muon import SingleDeviceMuon        from sophia import SophiaG        self.muon = SingleDeviceMuon(hidden, **muon_kw)        self.sophia = SophiaG(aux, **sophia_kw)    def zero_grad(self, set_to_none=True):        self.muon.zero_grad(set_to_none=set_to_none)        self.sophia.zero_grad(set_to_none=set_to_none)    @torch.no_grad()    def step(self, bs: int):        self.muon.step()        self.sophia.step(bs=bs)    @torch.no_grad()    def update_hessian(self):        self.sophia.update_hessian()def _build_optimizer(cfg: TrainConfig, model: TinyMoETransformer):    kind = (cfg.optimizer or 'adamw').lower()    if kind in {'adamw', 'adam'}:        optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)        sophia_meta = None        return optim, sophia_meta    if kind in {'muon_sophia', 'muon+sophia', 'ms'}:        hidden, aux = _split_params_for_muon(model)        opt = _MuonSophia(            hidden, aux,            muon_kw=dict(lr=cfg.muon_lr, weight_decay=cfg.muon_wd, momentum=0.95),            sophia_kw=dict(lr=cfg.sophia_lr, betas=(cfg.sophia_b1, cfg.sophia_b2), rho=cfg.sophia_rho, weight_decay=cfg.sophia_wd),        )        sophia_meta = {'k_hess': int(cfg.sophia_k)}        return opt, sophia_meta    if kind in {'sophia', 'sophia_g', 'sophiag'}:        from sophia import SophiaG        opt = SophiaG(model.parameters(), lr=cfg.sophia_lr, betas=(cfg.sophia_b1, cfg.sophia_b2), rho=cfg.sophia_rho, weight_decay=cfg.sophia_wd)        sophia_meta = {'k_hess': int(cfg.sophia_k)}        return opt, sophia_meta    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")def train(cfg: TrainConfig):    torch.backends.cuda.matmul.allow_tf32 = True    torch.backends.cudnn.allow_tf32 = True    torch.manual_seed(cfg.seed)    np.random.seed(cfg.seed)    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')    # Data    eval_sequences = max(cfg.eval_take, cfg.eval_iters * cfg.batch_size)    data = HFStreamingBatcher(        dataset_name=cfg.dataset_name,        dataset_config=cfg.dataset_config,        tokenizer=tokenizer,        block_size=cfg.block_size,        device=device,        seed=cfg.seed,        text_key=cfg.text_key,        shuffle_buffer=cfg.shuffle_buffer,        eval_sequences=eval_sequences,    )    # Model    cfg.vocab_size = len(tokenizer)    model = TinyMoETransformer(        vocab_size=cfg.vocab_size,        n_layer=cfg.n_layer,        n_head=cfg.n_head,        n_kv_heads=cfg.n_kv_heads,        d_model=cfg.d_model,        block_size=cfg.block_size,        dropout=cfg.dropout,        bias=cfg.bias,        n_experts=cfg.n_experts,        capacity_factor=cfg.capacity_factor,        dropless=cfg.dropless,        load_balance_alpha=cfg.load_balance_alpha,        router_z_loss_coef=cfg.router_z_loss_coef,        attn_gate=cfg.attn_gate,        router_temperature=cfg.router_temp_init,        router_noise_std=cfg.router_noise_std_init,        router_noise_type=cfg.router_noise_type,        use_rope=cfg.use_rope,        rope_theta=cfg.rope_theta,        moe_grouped=cfg.moe_grouped,        qk_norm=cfg.qk_norm,        qk_norm_eps=cfg.qk_norm_eps,    ).to(device=device, dtype=torch.bfloat16)    total_params = model.num_parameters()    print(f"Model params: {total_params:,} ({total_params/1e6:.2f}M)")    print(f"Config: layers={cfg.n_layer}, d_model={cfg.d_model}, heads={cfg.n_head}, experts={cfg.n_experts}")    if cfg.compile and hasattr(torch, 'compile'):        try:            import torch._inductor.config as inductor_config            if hasattr(inductor_config.triton, 'cudagraph_skip_dynamic_graphs'):                inductor_config.triton.cudagraph_skip_dynamic_graphs = True            if hasattr(inductor_config.triton, 'cudagraph_dynamic_shape_warn_limit'):                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None            if hasattr(inductor_config, 'shape_padding'):                inductor_config.shape_padding = True        except Exception:            pass        try:            model = torch.compile(model, mode='max-autotune')        except Exception as e:            print(f"torch.compile failed ({e}); continuing without compile.")    optimizer, sophia_meta = _build_optimizer(cfg, model)    logger = WandBLogger(project=cfg.wandb_project, run_name=cfg.wandb_run_name)    logger.watch(model)    ckpt_dir = Path(cfg.checkpoint_dir)    ckpt_dir.mkdir(parents=True, exist_ok=True)    start_iter, best_val = _maybe_resume(model, optimizer, cfg)    tokens_cum = 0    t0 = time.time()    clip_cum = 0    for it in range(start_iter, cfg.max_iters + 1):        # LR schedule        lr = cosine_lr(it, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters, cfg.lr_decay_iters)        for pg in optimizer.param_groups:            pg['lr'] = lr        # Router dynamics schedule per-step        curr_temp = _anneal_linear(it, cfg.router_temp_init, cfg.router_temp_final, cfg.router_temp_anneal_iters)        curr_noise = _anneal_linear(it, cfg.router_noise_std_init, 0.0, cfg.router_noise_decay_iters)        for blk in model.h:            if hasattr(blk, 'moe'):                blk.moe.router.set_router_state(curr_temp, curr_noise)                blk.moe.router.noise_type = cfg.router_noise_type        optimizer.zero_grad(set_to_none=True)        total_loss = 0.0        last_router_metrics = None        me_list, sv_list = [], []        for _ in range(cfg.gradient_accumulation_steps):            x, y = data.get_batch('train', cfg.batch_size)            logits, loss = model(x, y)            loss = loss / max(1, cfg.gradient_accumulation_steps)            loss.backward()            total_loss += float(loss.item())            # Router stats from the last micro-step only            with torch.no_grad():                auxs, max_fracs, num_active, drops, tpm, ent = [], [], [], [], [], []                for blk in model.h:                    if hasattr(blk, 'moe') and hasattr(blk.moe, '_last_stats'):                        st = blk.moe._last_stats                        auxs.append(float(st['aux']))                        max_fracs.append(float(st['max_frac']))                        num_active.append(int(st['num_active']))                        drops.append(float(st['drop_frac']))                        tpm.append(float(st['top1_p_mean']))                        ent.append(float(st['entropy_mean']))                        if 'me' in st and 'served' in st:                            me_list.append(st['me'].float().cpu())                            sv_list.append(st['served'].float().cpu())                if auxs:                    last_router_metrics = {                        'router/aux_mean': sum(auxs) / len(auxs),                        'router/max_frac_mean': sum(max_fracs) / len(max_fracs),                        'router/max_frac_max': max(max_fracs),                        'router/active_min': min(num_active),                        'router/active_mean': sum(num_active) / len(num_active),                        'router/drop_frac_mean': sum(drops) / len(drops),                        'router/top1_p_mean': sum(tpm) / len(tpm),                        'router/entropy_mean': sum(ent) / len(ent),                    }                    last_router_metrics['router/collapsed'] = 1 if last_router_metrics['router/max_frac_max'] >= 0.90 or last_router_metrics['router/active_min'] <= 1 else 0        # Gradient clip        if cfg.grad_clip is not None and cfg.grad_clip > 0:            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)            if torch.isfinite(norm):                clip_cum += int(norm > cfg.grad_clip)        optimizer.step(bs=int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps))        # Logging every cfg.log_interval        if (it % cfg.log_interval) == 0 and it > start_iter:            step_tokens = int(cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps)            tokens_cum += step_tokens            tps = step_tokens * cfg.log_interval / max(1e-6, (time.time() - t0))            avg_loss = total_loss            metrics = {                'train/loss': avg_loss,                'lr': lr,                'router/temp': curr_temp,                'router/noise_std': curr_noise,                'grad/clipped_cum': int(clip_cum),                'train/tokens_b': tokens_cum / 1e9,            }            if last_router_metrics:                metrics.update(last_router_metrics)            # Aggregate and log per-expert demand vs served fractions            if me_list:                _torch = torch                me_avg = _torch.stack(me_list).mean(0)                sv_avg = _torch.stack(sv_list).mean(0)                for e in range(len(me_avg)):                    metrics[f'router/frac_dem_e{e}'] = float(me_avg[e])                    metrics[f'router/frac_srv_e{e}'] = float(sv_avg[e])                metrics['router/frac_dem_max'] = float(me_avg.max())                metrics['router/frac_srv_max'] = float(sv_avg.max())                metrics['router/active_dem'] = int((me_avg > 0).sum())                metrics['router/active_srv'] = int((sv_avg > 0).sum())                # Combined utilization plot (served vs demand across experts)                try:                    import wandb as _wb                    xs = list(range(len(me_avg)))                    ys = [[float(v) for v in sv_avg.tolist()], [float(v) for v in me_avg.tolist()]]                    util_plot = _wb.plot.line_series(xs=xs, ys=ys, keys=['served','demand'], title='Expert Utilization', xname='expert')                    metrics['router/util_plot'] = util_plot                    metrics['router/frac_srv_sum'] = float(sv_avg.sum())                    metrics['router/frac_dem_sum'] = float(me_avg.sum())                except Exception:                    pass            logger.log_metrics(metrics, step=it)            if last_router_metrics:                rf = last_router_metrics['router/max_frac_max']                na = last_router_metrics['router/active_min']                col = last_router_metrics['router/collapsed']                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s | r_max {rf:.2f} act_min {na} col {col}")            else:                print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.3e} | {tps:.0f} tok/s")            t0 = time.time()        # Eval + checkpoint        if (it % cfg.eval_interval) == 0 and it > start_iter:            val_loss = evaluate(model, data, cfg, cfg.eval_iters)            logger.log_metrics({'val/loss': val_loss, 'val/ppl': math.exp(min(20.0, val_loss))}, step=it)            print(f"eval | val_loss {val_loss:.4f}")            # save 'last'            _save_ckpt(ckpt_dir / 'last_moe_bf16.pt', model, optimizer, cfg, it, val_loss)            # save 'best' on improvement            if val_loss < best_val:                best_val = val_loss                _save_ckpt(ckpt_dir / 'best_moe_bf16.pt', model, optimizer, cfg, it, val_loss)        # Periodic checkpointing        if cfg.save_interval and it > start_iter and (it % cfg.save_interval) == 0:            tag = f"iter_{it:06d}.pt"            _save_ckpt(ckpt_dir / tag, model, optimizer, cfg, it, None)            # prune old checkpoints            if cfg.keep_checkpoints and cfg.keep_checkpoints > 0:                ckpts = sorted(ckpt_dir.glob('iter_*.pt'))                if len(ckpts) > cfg.keep_checkpoints:                    for p in ckpts[:len(ckpts) - cfg.keep_checkpoints]:                        try:                            p.unlink()                        except Exception:                            pass    logger.set_summary(best_val_loss=best_val, params=total_params)    logger.finish()def main():    import argparse    p = argparse.ArgumentParser(description='BF16 Tiny MoE Transformer trainer')    # Model    p.add_argument('--vocab_size', type=int, default=32768)    p.add_argument('--n_layer', type=int, default=10)    p.add_argument('--n_head', type=int, default=8)    p.add_argument('--n_kv_heads', type=int, default=None)    p.add_argument('--d_model', type=int, default=512)    p.add_argument('--n_experts', type=int, default=4)    p.add_argument('--block_size', type=int, default=2048)    p.add_argument('--dropout', type=float, default=0.0)    p.add_argument('--bias', action='store_true')    p.add_argument('--capacity_factor', type=float, default=1.25)    p.add_argument('--dropless', action='store_true')    p.add_argument('--no-dropless', dest='dropless', action='store_false')    p.set_defaults(dropless=True)    p.add_argument('--load_balance_alpha', type=float, default=0.05)    p.add_argument('--router_z_loss_coef', type=float, default=0.0)    p.add_argument('--attn_gate', type=str, default='none', choices=['none', 'sigmoid_head'])    p.add_argument('--use_rope', dest='use_rope', action='store_true')    p.add_argument('--no-use_rope', dest='use_rope', action='store_false')    p.set_defaults(use_rope=True)    p.add_argument('--rope_theta', type=float, default=10000.0)    # QK-Norm    p.add_argument('--qk_norm', action='store_true')    p.add_argument('--qk_norm_eps', type=float, default=1e-6)    # Router dynamics CLI    p.add_argument('--router_temp_init', type=float, default=1.5)    p.add_argument('--router_temp_final', type=float, default=1.0)    p.add_argument('--router_temp_anneal_iters', type=int, default=1000)    p.add_argument('--router_noise_std_init', type=float, default=0.5)    p.add_argument('--router_noise_decay_iters', type=int, default=1000)    p.add_argument('--router_noise_type', type=str, default='gumbel', choices=['gumbel', 'gaussian'])    p.add_argument('--moe_grouped', action='store_true')    # Train    p.add_argument('--device', type=str, default='cuda')    p.add_argument('--batch_size', type=int, default=8)    p.add_argument('--gradient_accumulation_steps', type=int, default=1)    p.add_argument('--max_iters', type=int, default=2000)    p.add_argument('--eval_interval', type=int, default=200)    p.add_argument('--eval_iters', type=int, default=50)    p.add_argument('--learning_rate', type=float, default=3e-4)    p.add_argument('--min_lr', type=float, default=3e-5)    p.add_argument('--weight_decay', type=float, default=0.1)    p.add_argument('--beta1', type=float, default=0.9)    p.add_argument('--beta2', type=float, default=0.95)    p.add_argument('--warmup_iters', type=int, default=500)    p.add_argument('--lr_decay_iters', type=int, default=2000)    p.add_argument('--grad_clip', type=float, default=1.0)    p.add_argument('--compile', action='store_true')    p.add_argument('--seed', type=int, default=1337)    # Optimizer    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw','muon_sophia','sophia'])    p.add_argument('--muon_lr', type=float, default=3e-2)    p.add_argument('--muon_wd', type=float, default=1e-2)    p.add_argument('--sophia_lr', type=float, default=1e-3)    p.add_argument('--sophia_b1', type=float, default=0.965)    p.add_argument('--sophia_b2', type=float, default=0.99)    p.add_argument('--sophia_rho', type=float, default=0.1)    p.add_argument('--sophia_wd', type=float, default=0.2)    p.add_argument('--sophia_k', type=int, default=10)    # Data    p.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu')    p.add_argument('--dataset_config', type=str, default='sample-10BT')    p.add_argument('--text_key', type=str, default='text')    p.add_argument('--shuffle_buffer', type=int, default=8192)    p.add_argument('--eval_take', type=int, default=512)    # IO    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_moe_bf16')    p.add_argument('--log_interval', type=int, default=10)    p.add_argument('--save_interval', type=int, default=0)    p.add_argument('--keep_checkpoints', type=int, default=3)    p.add_argument('--resume', type=str, default=None)    p.add_argument('--auto_resume', action='store_true')    # Logging    p.add_argument('--wandb_project', type=str, default='moe-bf16-experiments_v2')    p.add_argument('--wandb_run_name', type=str, default=None)    args = p.parse_args()    cfg = TrainConfig(**vars(args))    d = cfg.d_model    params_per_expert = 8 * (d ** 2)    print(f"Estimated params/expert (SwiGLU): ~{params_per_expert/1e6:.2f}M for d={d}")    train(cfg)if __name__ == '__main__':        try:

