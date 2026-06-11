#!/bin/bash
# v12.8 phase B: low-LR full-model continuation after relative-bias warmup.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
BASE_CKPT="${V128_BASE_CKPT:-/home/laure/alphaxiang/training_runs/run_019a_v128_relbias_warmup/latest.pt}"
OUT_DIR="${V128_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_019b_v128_relbias_full}"
DEVICE="${V128_TRAIN_DEVICE:-cuda:0}"
MAX_STEPS="${V128_MAX_STEPS:-302000}"
LR="${V128_LR:-2e-5}"

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
    --use-2d-relative-attention-bias \
    --learning-rate "$LR" \
    --warmup-steps 0 \
    --max-steps "$MAX_STEPS" \
    --lr-schedule-max-steps "$MAX_STEPS" \
    --save-interval-steps 1000 \
    --snapshot-interval-steps 1000 \
    --eval-interval-steps 1000 \
    --log-interval-steps 100 \
    --micro-batch-size 1024 \
    --grad-accum-steps 1 \
    --wdl-loss-weight 1.0 \
    --value-loss-weight 0.5 \
    --wdl-value-consistency-weight 0.02 \
    --value-target-scale 0.9 \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight 0.0 \
    --seed 1281902
