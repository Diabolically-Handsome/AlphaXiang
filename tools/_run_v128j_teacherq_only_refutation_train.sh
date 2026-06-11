#!/bin/bash
# v12.8J train: full-Pika refutation as teacher-Q-only regularizer.
# Ordinary policy/value/WDL losses are disabled so the small d5 slice cannot
# rewrite the global-strategy style; it can only nudge candidate ordering while
# the v12.8E anchor preserves behavior.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
RESUME_CKPT="${V128J_RESUME_CKPT:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt}"
ANCHOR_CKPT="${V128J_ANCHOR_CKPT:-$RESUME_CKPT}"
REFUTE_DIR="${V128J_REFUTE_DIR:-/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses}"
OUT_DIR="${V128J_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_019l_v128j_teacherq_only_refute}"
DEVICE="${V128J_TRAIN_DEVICE:-cuda:0}"
MAX_STEPS="${V128J_MAX_STEPS:-297500}"
LR="${V128J_LR:-1e-5}"
MICRO_BATCH="${V128J_MICRO_BATCH:-128}"
STRATEGY_TOKENS="${V128J_GLOBAL_STRATEGY_TOKENS:-6}"
DIR_RATIOS="${V128J_DIR_RATIOS:-0.70 0.20 0.10}"
REPLAY_BUFFER_SIZE="${V128J_REPLAY_BUFFER_SIZE:-70000}"
ANCHOR_POLICY_KL="${V128J_ANCHOR_POLICY_KL:-0.35}"
ANCHOR_VALUE_MSE="${V128J_ANCHOR_VALUE_MSE:-0.08}"
TEACHER_Q_WEIGHT="${V128J_TEACHER_Q_WEIGHT:-0.12}"
TEACHER_Q_TEMP="${V128J_TEACHER_Q_TEMP:-80.0}"

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
    --reset-selfplay-ingest-state-on-resume \
    --replay-buffer-size "$REPLAY_BUFFER_SIZE" \
    --selfplay-dirs \
        /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
        /home/laure/alphaxiang/v126_day3_d4_slice \
        "$REFUTE_DIR" \
    --selfplay-dir-sampling-ratios $DIR_RATIOS \
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
    --save-interval-steps 250 \
    --snapshot-interval-steps 250 \
    --eval-interval-steps 250 \
    --log-interval-steps 50 \
    --micro-batch-size "$MICRO_BATCH" \
    --grad-accum-steps 1 \
    --samples-per-unit 64 \
    --wdl-loss-weight 0.0 \
    --policy-loss-weight 0.0 \
    --value-loss-weight 0.0 \
    --wdl-value-consistency-weight 0.0 \
    --value-target-scale 0.9 \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight "$TEACHER_Q_WEIGHT" \
    --teacher-q-temperature-cp "$TEACHER_Q_TEMP" \
    --seed 1281912
