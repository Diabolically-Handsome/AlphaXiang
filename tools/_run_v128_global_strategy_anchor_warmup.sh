#!/bin/bash
# v12.8 phase E: global strategic-attention adapter with v12.6 behavior anchoring.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
BASE_CKPT="${V128_BASE_CKPT:-/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt}"
ANCHOR_CKPT="${V128_ANCHOR_CKPT:-$BASE_CKPT}"
OUT_DIR="${V128_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor}"
DEVICE="${V128_TRAIN_DEVICE:-cuda:0}"
MAX_STEPS="${V128_MAX_STEPS:-297000}"
LR="${V128_LR:-5e-5}"
MICRO_BATCH="${V128_MICRO_BATCH:-192}"
STRATEGY_TOKENS="${V128_GLOBAL_STRATEGY_TOKENS:-6}"
ANCHOR_POLICY_KL="${V128_ANCHOR_POLICY_KL:-0.05}"
ANCHOR_VALUE_MSE="${V128_ANCHOR_VALUE_MSE:-0.02}"

cd "$REPO"

"$PY" xiangqi_train.py \
    --resume-path "$BASE_CKPT" \
    --reset-optimizer-on-resume \
    --selfplay-dirs \
        /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
        /home/laure/alphaxiang/v126_day3_d4_slice \
    --output-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --foreground \
    --use-global-strategic-attention \
    --num-global-strategy-tokens "$STRATEGY_TOKENS" \
    --train-only-transformer-adapters \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --anchor-policy-kl-weight "$ANCHOR_POLICY_KL" \
    --anchor-value-mse-weight "$ANCHOR_VALUE_MSE" \
    --learning-rate "$LR" \
    --warmup-steps 0 \
    --max-steps "$MAX_STEPS" \
    --lr-schedule-max-steps "$MAX_STEPS" \
    --save-interval-steps 500 \
    --snapshot-interval-steps 500 \
    --eval-interval-steps 500 \
    --log-interval-steps 100 \
    --micro-batch-size "$MICRO_BATCH" \
    --grad-accum-steps 1 \
    --wdl-loss-weight 1.0 \
    --value-loss-weight 0.5 \
    --wdl-value-consistency-weight 0.02 \
    --value-target-scale 0.9 \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight 0.0 \
    --seed 1281906
