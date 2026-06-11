#!/bin/bash
# v12.8I train: use full-Pika refutation as a low-frequency balanced
# regularizer instead of oversampling it into the whole replay stream.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
RESUME_CKPT="${V128I_RESUME_CKPT:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt}"
ANCHOR_CKPT="${V128I_ANCHOR_CKPT:-$RESUME_CKPT}"
REFUTE_DIR="${V128I_REFUTE_DIR:-/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses}"
OUT_DIR="${V128I_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_019k_v128i_balanced_refute_anchor_e}"
DEVICE="${V128I_TRAIN_DEVICE:-cuda:0}"
MAX_STEPS="${V128I_MAX_STEPS:-297500}"
LR="${V128I_LR:-2e-5}"
MICRO_BATCH="${V128I_MICRO_BATCH:-128}"
STRATEGY_TOKENS="${V128I_GLOBAL_STRATEGY_TOKENS:-6}"
DIR_RATIOS="${V128I_DIR_RATIOS:-0.74 0.20 0.06}"
ANCHOR_POLICY_KL="${V128I_ANCHOR_POLICY_KL:-0.20}"
ANCHOR_VALUE_MSE="${V128I_ANCHOR_VALUE_MSE:-0.04}"
TEACHER_Q_WEIGHT="${V128I_TEACHER_Q_WEIGHT:-0.08}"
TEACHER_Q_TEMP="${V128I_TEACHER_Q_TEMP:-70.0}"
POLICY_ORACLE_ALPHA="${V128I_POLICY_ORACLE_ALPHA:-0.03}"

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
    --wdl-loss-weight 1.0 \
    --value-loss-weight 0.5 \
    --wdl-value-consistency-weight 0.02 \
    --value-target-scale 0.9 \
    --policy-oracle-alpha "$POLICY_ORACLE_ALPHA" \
    --teacher-q-loss-weight "$TEACHER_Q_WEIGHT" \
    --teacher-q-temperature-cp "$TEACHER_Q_TEMP" \
    --seed 1281911
