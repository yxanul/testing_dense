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




root@cec3dd147d8b:/workspace/testing_dense# python train_experimental.py --device cuda --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --n_experts 8 --dropless --load_balance_alpha 0.12 --router_z_loss_coef 1e-3 --rout
er_temp_init 2.2 --router_temp_final 1.3 --router_temp_anneal_iters 5000 --router_noise_std_init 0.9 --router_noise_deca
y_iters 5000 --router_noise_type gumbel --attn_gate sigmoid_head --qk_norm --use_rope --batch_size 12 --gradient_accumul
ation_steps 12 --max_iters 8000 --lr_decay_iters 8000 --warmup_iters 2000 --eval_interval 200 --eval_iters 50 --wandb_pr
oject moe-bf16-experiments_v2 --optimizer muon_sophia --muon_lr 1e-2 --muon_wd 1e-2 --sophia_lr 6e-4 --sophia_b1 0.965 -
-sophia_b2 0.99 --sophia_rho 0.1 --sophia_wd 0.2 --sophia_k 10 --log_interval 20
Estimated params/expert (SwiGLU): ~2.10M for d=512
Resolving data files: 100%|███████████████████████████████████████████████████████| 2410/2410 [00:00<00:00, 7498.67it/s]
Model params: 202,952,528 (202.95M)
Config: layers=10, d_model=512, heads=8, experts=8
wandb: Currently logged in as: davidfranco2300 (davidfranco2300-other) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.4
wandb: Run data is saved locally in /workspace/testing_dense/wandb/run-20250917_161337-n9dmr5eq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run neat-sky-8
wandb: ⭐️ View project at https://wandb.ai/davidfranco2300-other/moe-bf16-experiments_v2
wandb: 🚀 View run at https://wandb.ai/davidfranco2300-other/moe-bf16-experiments_v2/runs/n9dmr5eq
Resolving data files: 100%|███████████████████████████████████████████████████████| 2410/2410 [00:00<00:00, 7528.14it/s]
iter     20 | loss 10.6008 | lr 3.150e-06 | 77154 tok/s | r_max 0.16 act_min 8 col 0
iter     40 | loss 10.5822 | lr 6.150e-06 | 82154 tok/s | r_max 0.18 act_min 8 col 0
iter     60 | loss 10.5143 | lr 9.150e-06 | 82352 tok/s | r_max 0.17 act_min 8 col 0
iter     80 | loss 10.4119 | lr 1.215e-05 | 81744 tok/s | r_max 0.16 act_min 8 col 0
iter    100 | loss 10.2737 | lr 1.515e-05 | 80947 tok/s | r_max 0.17 act_min 8 col 0
iter    120 | loss 9.9865 | lr 1.815e-05 | 81941 tok/s | r_max 0.17 act_min 8 col 0
iter    140 | loss 9.6744 | lr 2.115e-05 | 82984 tok/s | r_max 0.18 act_min 8 col 0
iter    160 | loss 9.3559 | lr 2.415e-05 | 82793 tok/s | r_max 0.19 act_min 8 col 0
iter    180 | loss 9.0216 | lr 2.715e-05 | 82756 tok/s | r_max 0.20 act_min 8 col 0
iter    200 | loss 8.7454 | lr 3.015e-05 | 81286 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 8.7386
iter    220 | loss 8.2175 | lr 3.315e-05 | 77393 tok/s | r_max 0.18 act_min 8 col 0
iter    240 | loss 7.8458 | lr 3.615e-05 | 82411 tok/s | r_max 0.19 act_min 8 col 0
iter    260 | loss 7.5576 | lr 3.915e-05 | 82167 tok/s | r_max 0.21 act_min 8 col 0
iter    280 | loss 7.2938 | lr 4.215e-05 | 81444 tok/s | r_max 0.22 act_min 8 col 0
iter    300 | loss 7.0860 | lr 4.515e-05 | 82690 tok/s | r_max 0.22 act_min 8 col 0
iter    320 | loss 6.9692 | lr 4.815e-05 | 82277 tok/s | r_max 0.20 act_min 8 col 0
iter    340 | loss 6.7872 | lr 5.115e-05 | 82185 tok/s | r_max 0.19 act_min 8 col 0
iter    360 | loss 6.6896 | lr 5.415e-05 | 82817 tok/s | r_max 0.18 act_min 8 col 0
iter    380 | loss 6.5048 | lr 5.715e-05 | 81297 tok/s | r_max 0.16 act_min 8 col 0
iter    400 | loss 6.3674 | lr 6.015e-05 | 82586 tok/s | r_max 0.16 act_min 8 col 0
eval | val_loss 6.3619
iter    420 | loss 6.2578 | lr 6.315e-05 | 77821 tok/s | r_max 0.16 act_min 8 col 0
iter    440 | loss 6.0900 | lr 6.615e-05 | 82761 tok/s | r_max 0.15 act_min 8 col 0
iter    460 | loss 6.0383 | lr 6.915e-05 | 82767 tok/s | r_max 0.16 act_min 8 col 0
iter    480 | loss 5.9089 | lr 7.215e-05 | 81759 tok/s | r_max 0.15 act_min 8 col 0
iter    500 | loss 5.8857 | lr 7.515e-05 | 83090 tok/s | r_max 0.16 act_min 8 col 0
iter    520 | loss 5.8606 | lr 7.815e-05 | 82054 tok/s | r_max 0.18 act_min 8 col 0
iter    540 | loss 5.7870 | lr 8.115e-05 | 83023 tok/s | r_max 0.17 act_min 8 col 0
iter    560 | loss 5.7229 | lr 8.415e-05 | 82800 tok/s | r_max 0.17 act_min 8 col 0
iter    580 | loss 5.6452 | lr 8.715e-05 | 81690 tok/s | r_max 0.18 act_min 8 col 0
iter    600 | loss 5.5309 | lr 9.015e-05 | 82689 tok/s | r_max 0.19 act_min 8 col 0
eval | val_loss 5.5051
iter    620 | loss 5.5293 | lr 9.315e-05 | 77575 tok/s | r_max 0.18 act_min 8 col 0
iter    640 | loss 5.3708 | lr 9.615e-05 | 82863 tok/s | r_max 0.20 act_min 8 col 0
iter    660 | loss 5.4073 | lr 9.915e-05 | 82785 tok/s | r_max 0.20 act_min 8 col 0
iter    680 | loss 5.3128 | lr 1.021e-04 | 81126 tok/s | r_max 0.19 act_min 8 col 0
iter    700 | loss 5.2661 | lr 1.051e-04 | 82621 tok/s | r_max 0.21 act_min 8 col 0
iter    720 | loss 5.2517 | lr 1.081e-04 | 82925 tok/s | r_max 0.22 act_min 8 col 0
iter    740 | loss 5.1910 | lr 1.111e-04 | 82275 tok/s | r_max 0.26 act_min 8 col 0
iter    760 | loss 5.2107 | lr 1.141e-04 | 80689 tok/s | r_max 0.19 act_min 8 col 0
iter    780 | loss 5.0518 | lr 1.171e-04 | 82746 tok/s | r_max 0.21 act_min 8 col 0
iter    800 | loss 5.0800 | lr 1.201e-04 | 82580 tok/s | r_max 0.22 act_min 8 col 0
eval | val_loss 4.9537
iter    820 | loss 5.0149 | lr 1.231e-04 | 77993 tok/s | r_max 0.20 act_min 8 col 0
iter    840 | loss 4.9895 | lr 1.261e-04 | 82760 tok/s | r_max 0.20 act_min 8 col 0
iter    860 | loss 5.0388 | lr 1.291e-04 | 81367 tok/s | r_max 0.23 act_min 8 col 0
iter    880 | loss 4.9285 | lr 1.321e-04 | 82412 tok/s | r_max 0.22 act_min 8 col 0
iter    900 | loss 4.9009 | lr 1.352e-04 | 82825 tok/s | r_max 0.28 act_min 8 col 0
iter    920 | loss 4.8467 | lr 1.381e-04 | 82409 tok/s | r_max 0.26 act_min 8 col 0
iter    940 | loss 4.8412 | lr 1.411e-04 | 82289 tok/s | r_max 0.27 act_min 8 col 0
iter    960 | loss 4.8114 | lr 1.442e-04 | 80651 tok/s | r_max 0.25 act_min 8 col 0
iter    980 | loss 4.6931 | lr 1.471e-04 | 82201 tok/s | r_max 0.25 act_min 8 col 0
iter   1000 | loss 4.7359 | lr 1.501e-04 | 83305 tok/s | r_max 0.26 act_min 8 col 0
eval | val_loss 4.6267
iter   1020 | loss 4.6994 | lr 1.531e-04 | 77031 tok/s | r_max 0.24 act_min 8 col 0
iter   1040 | loss 4.7930 | lr 1.561e-04 | 82439 tok/s | r_max 0.23 act_min 8 col 0
iter   1060 | loss 4.4942 | lr 1.591e-04 | 80827 tok/s | r_max 0.23 act_min 8 col 0
iter   1080 | loss 4.5923 | lr 1.621e-04 | 82575 tok/s | r_max 0.22 act_min 8 col 0
iter   1100 | loss 4.6684 | lr 1.652e-04 | 82850 tok/s | r_max 0.26 act_min 8 col 0
iter   1120 | loss 4.5733 | lr 1.681e-04 | 81748 tok/s | r_max 0.29 act_min 8 col 0
iter   1140 | loss 4.4835 | lr 1.711e-04 | 81012 tok/s | r_max 0.26 act_min 8 col 0
iter   1160 | loss 4.6147 | lr 1.741e-04 | 82326 tok/s | r_max 0.22 act_min 8 col 0
iter   1180 | loss 4.5002 | lr 1.771e-04 | 82577 tok/s | r_max 0.27 act_min 8 col 0
iter   1200 | loss 4.5042 | lr 1.801e-04 | 82300 tok/s | r_max 0.25 act_min 8 col 0
eval | val_loss 4.3896
iter   1220 | loss 4.5157 | lr 1.831e-04 | 77292 tok/s | r_max 0.22 act_min 8 col 0
iter   1240 | loss 4.4554 | lr 1.861e-04 | 80559 tok/s | r_max 0.28 act_min 8 col 0
iter   1260 | loss 4.4411 | lr 1.891e-04 | 82965 tok/s | r_max 0.25 act_min 8 col 0
iter   1280 | loss 4.4618 | lr 1.921e-04 | 81716 tok/s | r_max 0.26 act_min 8 col 0
iter   1300 | loss 4.4209 | lr 1.952e-04 | 82683 tok/s | r_max 0.26 act_min 8 col 0
iter   1320 | loss 4.3999 | lr 1.981e-04 | 82274 tok/s | r_max 0.27 act_min 8 col 0
iter   1340 | loss 4.3366 | lr 2.011e-04 | 82141 tok/s | r_max 0.25 act_min 8 col 0
iter   1360 | loss 4.3724 | lr 2.041e-04 | 82600 tok/s | r_max 0.25 act_min 8 col 0
iter   1380 | loss 4.3805 | lr 2.071e-04 | 82909 tok/s | r_max 0.25 act_min 8 col 0
iter   1400 | loss 4.3524 | lr 2.101e-04 | 82604 tok/s | r_max 0.24 act_min 8 col 0
eval | val_loss 4.1990
iter   1420 | loss 4.2645 | lr 2.131e-04 | 77361 tok/s | r_max 0.28 act_min 8 col 0
iter   1440 | loss 4.2707 | lr 2.161e-04 | 81545 tok/s | r_max 0.26 act_min 8 col 0
iter   1460 | loss 4.2221 | lr 2.191e-04 | 82125 tok/s | r_max 0.27 act_min 8 col 0
iter   1480 | loss 4.3194 | lr 2.221e-04 | 82443 tok/s | r_max 0.27 act_min 8 col 0
iter   1500 | loss 4.2171 | lr 2.251e-04 | 81873 tok/s | r_max 0.25 act_min 8 col 0
iter   1520 | loss 4.1992 | lr 2.281e-04 | 80927 tok/s | r_max 0.25 act_min 8 col 0
iter   1540 | loss 4.2949 | lr 2.311e-04 | 82281 tok/s | r_max 0.28 act_min 8 col 0
iter   1560 | loss 4.1975 | lr 2.341e-04 | 82136 tok/s | r_max 0.25 act_min 8 col 0
iter   1580 | loss 4.2299 | lr 2.371e-04 | 82271 tok/s | r_max 0.30 act_min 8 col 0
iter   1600 | loss 4.1753 | lr 2.401e-04 | 82319 tok/s | r_max 0.21 act_min 8 col 0
eval | val_loss 4.0625
iter   1620 | loss 4.1839 | lr 2.431e-04 | 76274 tok/s | r_max 0.26 act_min 8 col 0
iter   1640 | loss 4.2026 | lr 2.461e-04 | 81881 tok/s | r_max 0.24 act_min 8 col 0
iter   1660 | loss 4.1830 | lr 2.491e-04 | 82401 tok/s | r_max 0.26 act_min 8 col 0
iter   1680 | loss 4.2155 | lr 2.521e-04 | 82309 tok/s | r_max 0.28 act_min 8 col 0
iter   1700 | loss 4.1569 | lr 2.551e-04 | 82146 tok/s | r_max 0.29 act_min 8 col 0
iter   1720 | loss 4.0801 | lr 2.582e-04 | 81983 tok/s | r_max 0.28 act_min 8 col 0
iter   1740 | loss 4.1342 | lr 2.611e-04 | 82677 tok/s | r_max 0.27 act_min 8 col 0
iter   1760 | loss 4.0314 | lr 2.641e-04 | 82327 tok/s | r_max 0.25 act_min 8 col 0
iter   1780 | loss 4.0902 | lr 2.672e-04 | 82207 tok/s | r_max 0.26 act_min 8 col 0
iter   1800 | loss 4.0889 | lr 2.701e-04 | 83259 tok/s | r_max 0.30 act_min 8 col 0
eval | val_loss 3.9582
iter   1820 | loss 3.9305 | lr 2.731e-04 | 77020 tok/s | r_max 0.30 act_min 8 col 0
iter   1840 | loss 4.1403 | lr 2.761e-04 | 81441 tok/s | r_max 0.25 act_min 8 col 0
iter   1860 | loss 4.1151 | lr 2.791e-04 | 81685 tok/s | r_max 0.30 act_min 8 col 0
iter   1880 | loss 4.0210 | lr 2.821e-04 | 82265 tok/s | r_max 0.31 act_min 8 col 0
iter   1900 | loss 4.0901 | lr 2.851e-04 | 81223 tok/s | r_max 0.21 act_min 8 col 0
iter   1920 | loss 4.0563 | lr 2.881e-04 | 82360 tok/s | r_max 0.27 act_min 8 col 0
iter   1940 | loss 4.0599 | lr 2.911e-04 | 81881 tok/s | r_max 0.26 act_min 8 col 0
iter   1960 | loss 4.0082 | lr 2.941e-04 | 82361 tok/s | r_max 0.30 act_min 8 col 0
iter   1980 | loss 4.0254 | lr 2.971e-04 | 82088 tok/s | r_max 0.24 act_min 8 col 0
iter   2000 | loss 4.0689 | lr 3.000e-04 | 80869 tok/s | r_max 0.28 act_min 8 col 0
eval | val_loss 3.8795
iter   2020 | loss 4.0290 | lr 3.000e-04 | 76871 tok/s | r_max 0.32 act_min 8 col 0
iter   2040 | loss 3.9535 | lr 3.000e-04 | 82706 tok/s | r_max 0.30 act_min 8 col 0
iter   2060 | loss 3.9188 | lr 2.999e-04 | 82207 tok/s | r_max 0.34 act_min 8 col 0
iter   2080 | loss 3.9660 | lr 2.999e-04 | 82831 tok/s | r_max 0.24 act_min 8 col 0
iter   2100 | loss 3.9953 | lr 2.998e-04 | 81381 tok/s | r_max 0.22 act_min 8 col 0
iter   2120 | loss 3.9679 | lr 2.997e-04 | 82802 tok/s | r_max 0.28 act_min 8 col 0
iter   2140 | loss 3.9593 | lr 2.996e-04 | 82276 tok/s | r_max 0.24 act_min 8 col 0
iter   2160 | loss 3.9930 | lr 2.995e-04 | 82469 tok/s | r_max 0.29 act_min 8 col 0
iter   2180 | loss 3.9888 | lr 2.994e-04 | 82370 tok/s | r_max 0.27 act_min 8 col 0
iter   2200 | loss 3.9387 | lr 2.993e-04 | 80813 tok/s | r_max 0.28 act_min 8 col 0
eval | val_loss 3.8117
iter   2220 | loss 4.0041 | lr 2.991e-04 | 76812 tok/s | r_max 0.27 act_min 8 col 0
iter   2240 | loss 3.9442 | lr 2.989e-04 | 81600 tok/s | r_max 0.33 act_min 8 col 0
iter   2260 | loss 3.8991 | lr 2.988e-04 | 81994 tok/s | r_max 0.29 act_min 8 col 0
iter   2280 | loss 3.9760 | lr 2.986e-04 | 81231 tok/s | r_max 0.28 act_min 8 col 0
iter   2300 | loss 4.1271 | lr 2.983e-04 | 82254 tok/s | r_max 0.30 act_min 8 col 0
iter   2320 | loss 3.8980 | lr 2.981e-04 | 82482 tok/s | r_max 0.28 act_min 8 col 0
iter   2340 | loss 3.9738 | lr 2.979e-04 | 83204 tok/s | r_max 0.33 act_min 8 col 0
iter   2360 | loss 3.8969 | lr 2.976e-04 | 82075 tok/s | r_max 0.28 act_min 8 col 0
iter   2380 | loss 3.9222 | lr 2.973e-04 | 80525 tok/s | r_max 0.19 act_min 8 col 0
iter   2400 | loss 3.8842 | lr 2.970e-04 | 82891 tok/s | r_max 0.25 act_min 8 col 0
eval | val_loss 3.7640
iter   2420 | loss 3.8498 | lr 2.967e-04 | 77157 tok/s | r_max 0.30 act_min 8 col 0
iter   2440 | loss 3.7504 | lr 2.964e-04 | 82321 tok/s | r_max 0.31 act_min 8 col 0
iter   2460 | loss 4.0036 | lr 2.961e-04 | 82627 tok/s | r_max 0.30 act_min 8 col 0
iter   2480 | loss 3.9195 | lr 2.958e-04 | 81230 tok/s | r_max 0.28 act_min 8 col 0
iter   2500 | loss 3.9551 | lr 2.954e-04 | 82597 tok/s | r_max 0.34 act_min 8 col 0


python train_experimental.py --device cuda --dataset_name HuggingFaceFW/fineweb-edu --dataset_config sample-10BT --n_experts 8 --dropless --load_balance_alpha 0.12 --router_z_loss_coef 1e-3 --router_temp_init 2.2 --router_temp_final 1.3 --router_temp_anneal_iters 5000 --router_noise_std_init 0.9 --router_noise_decay_iters 5000 --router_noise_type gumbel --attn_gate sigmoid_head --qk_norm --use_rope --batch_size 8 --gradient_accumulation_steps 12 --max_iters 8000 --lr_decay_iters 8000 --warmup_iters 1000 --eval_interval 200 --eval_iters 50 --wandb_project moe-bf16-experiments_v2 --optimizer adamw --log_interval 20 --compile --learning_rate 6e-4 --n_layer 16