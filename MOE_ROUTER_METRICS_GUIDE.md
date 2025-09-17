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


python train_experimental.py --device cuda --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --n_experts 8 --dropless --load_balance_alpha 0.12 --router_z_loss_coef 1e-3 --router_temp_init 2.2 --router_temp_final 1.3 --router_temp_anneal_iters 5000 --router_noise_std_init 0.9 --router_noise_decay_iters 5000 --router_noise_type gumbel --attn_gate sigmoid_head --qk_norm --use_rope --batch_size 12 --gradient_accumulation_steps 12 --learning_rate 8e-4 --max_iters 8000 --lr_decay_iters 8000 --warmup_iters 1000 --eval_interval 200 --eval_iters 50 --wandb_project moe-bf16-experiments_v2 --optimizer muon_sophia --muon_lr 3e-2 --muon_wd 1e-2 --sophia_lr 1e-3 --sophia_b1 0.965 --sophia_b2 0.99 --sophia_rho 0.1 --sophia_wd 0.2 --sophia_k 10
