@echo off
REM Comparison experiments for Gated Attention vs Baseline

echo ==========================================
echo GPT-2 GATED ATTENTION EXPERIMENTS
echo ==========================================

REM Common arguments for quick testing
set COMMON_ARGS=--batch_size 12 --sequence_length 2048 --num_train_steps 10000 --eval_interval 500

echo.
echo 1. Training BASELINE model (no gated attention)...
echo ==========================================
python train_gated.py --no_gated_attention %COMMON_ARGS%

echo.
echo 2. Training GATED ATTENTION model...
echo ==========================================
python train_gated.py --use_gated_attention %COMMON_ARGS%

echo.
echo 3. Training GATED + QK_NORM model...
echo ==========================================
python train_gated.py --use_gated_attention --use_qk_norm %COMMON_ARGS%

echo.
echo ==========================================
echo ALL EXPERIMENTS COMPLETED
echo Check WandB for comparisons
echo ==========================================