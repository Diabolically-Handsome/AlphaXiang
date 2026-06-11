#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"

PY="/home/laure/alphaxiang/venv_nospace/bin/python"
BASE="/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt"
OUT="/home/laure/alphaxiang/training_runs/run_041a_v14b_cnn_policy_residual_from031a19000"

exec "$PY" -u xiangqi_train.py \
  --foreground \
  --model-preset v14b_200m_cnn_policy_residual \
  --resume-path "$BASE" \
  --reset-optimizer-on-resume \
  --reset-selfplay-ingest-state-on-resume \
  --anchor-checkpoint "$BASE" \
  --output-dir "$OUT" \
  --selfplay-dirs \
    /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
    /home/laure/alphaxiang/v126_day3_d4_slice \
    /home/laure/alphaxiang/v131_weekend_detox/clean_d5 \
    /home/laure/alphaxiang/v133_p6_fullpika_round2_black_d5_verified_blunders_teacherq_d18 \
    /home/laure/alphaxiang/v133_p6_fullpika_round3_black_d6_verified_blunders_teacherq_d20 \
    /home/laure/alphaxiang/v133_p6_fullpika_round4_black_d6_verified_blunders_teacherq_d20 \
  --selfplay-dir-sampling-ratios 0.65 0.10 0.05 0.07 0.06 0.07 \
  --device cuda:0 \
  --replay-buffer-size 70000 \
  --micro-batch-size 64 \
  --grad-accum-steps 1 \
  --cpu-sampler-workers 16 \
  --cpu-prefetch-batches 16 \
  --learning-rate 1e-6 \
  --warmup-steps 0 \
  --max-steps 20000 \
  --save-interval-steps 250 \
  --snapshot-interval-steps 250 \
  --eval-interval-steps 500 \
  --log-interval-steps 20 \
  --train-only-cnn-policy-residual-adapter \
  --anchor-policy-kl-weight 3.0 \
  --anchor-value-mse-weight 0.25
