#!/bin/bash
# v12.8H train: continue the best anchored global-strategy checkpoint on
# full-Pika d5 refutation data.  Model capacity stays unchanged.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
RESUME_CKPT="${V128H_RESUME_CKPT:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt}"
ANCHOR_CKPT="${V128H_ANCHOR_CKPT:-/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt}"
REFUTE_DIR="${V128H_REFUTE_DIR:-/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses_x16}"
OUT_DIR="${V128H_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_019i_v128h_fullpika_refutation_x16}"
DEVICE="${V128H_TRAIN_DEVICE:-cuda:0}"
MAX_STEPS="${V128H_MAX_STEPS:-298000}"
LR="${V128H_LR:-3e-5}"
MICRO_BATCH="${V128H_MICRO_BATCH:-96}"
STRATEGY_TOKENS="${V128H_GLOBAL_STRATEGY_TOKENS:-6}"
ANCHOR_POLICY_KL="${V128H_ANCHOR_POLICY_KL:-0.07}"
ANCHOR_VALUE_MSE="${V128H_ANCHOR_VALUE_MSE:-0.02}"
TEACHER_Q_WEIGHT="${V128H_TEACHER_Q_WEIGHT:-0.30}"
TEACHER_Q_TEMP="${V128H_TEACHER_Q_TEMP:-60.0}"
POLICY_ORACLE_ALPHA="${V128H_POLICY_ORACLE_ALPHA:-0.15}"

if [ ! -f "$RESUME_CKPT" ]; then
    echo "missing resume checkpoint: $RESUME_CKPT" >&2
    exit 1
fi
if [ ! -f "$ANCHOR_CKPT" ]; then
    echo "missing anchor checkpoint: $ANCHOR_CKPT" >&2
    exit 1
fi
if [ ! -f "$REFUTE_DIR/manifest.json" ]; then
    echo "missing refutation data manifest: $REFUTE_DIR/manifest.json" >&2
    exit 1
fi

cd "$REPO"

"$PY" xiangqi_train.py \
    --resume-path "$RESUME_CKPT" \
    --reset-optimizer-on-resume \
    --selfplay-dirs \
        /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
        /home/laure/alphaxiang/v126_day3_d4_slice \
        "$REFUTE_DIR" \
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
    --policy-oracle-alpha "$POLICY_ORACLE_ALPHA" \
    --teacher-q-loss-weight "$TEACHER_Q_WEIGHT" \
    --teacher-q-temperature-cp "$TEACHER_Q_TEMP" \
    --seed 1281909
