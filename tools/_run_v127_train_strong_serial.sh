#!/bin/bash
# Run the v12.7 Strong regret finetune arm in serial after the Light arm frees cuda:0.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
DATA_ROOT="/home/laure/alphaxiang/v127_regret_data"
FAILURE_DIR="$DATA_ROOT/failure_d4d5"
ROOT_GUARD_DIR="$DATA_ROOT/root_guard_events"
EXISTING_SLICE="/home/laure/alphaxiang/v126_day3_d4_slice"
OUT_DIR="${V127_STRONG_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018b_v127_regret_strong_serial}"
DEVICE="${V127_STRONG_DEVICE:-cuda:0}"
MICRO_BATCH_SIZE="${V127_STRONG_MICRO_BATCH_SIZE:-1024}"
GRAD_ACCUM_STEPS="${V127_STRONG_GRAD_ACCUM_STEPS:-1}"

require_manifest() {
    local run_dir="$1"
    if [ ! -f "$run_dir/manifest.json" ]; then
        echo "missing manifest: $run_dir/manifest.json" >&2
        exit 1
    fi
}

require_manifest "$FAILURE_DIR"
require_manifest "$ROOT_GUARD_DIR"
require_manifest "$EXISTING_SLICE"

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" xiangqi_train.py \
    --resume-path "$CKPT" \
    --reset-optimizer-on-resume \
    --selfplay-dirs "$FAILURE_DIR" "$ROOT_GUARD_DIR" "$EXISTING_SLICE" \
    --output-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --foreground \
    --learning-rate 5e-5 \
    --warmup-steps 0 \
    --max-steps 304000 \
    --lr-schedule-max-steps 304000 \
    --snapshot-interval-steps 2000 \
    --save-interval-steps 2000 \
    --eval-interval-steps 2000 \
    --log-interval-steps 100 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --teacher-q-loss-weight 0.25 \
    --teacher-q-temperature-cp 80 \
    --policy-oracle-alpha 0.40 \
    --seed 1270181 \
    2>&1 | tee "$OUT_DIR/train_stdout.log"

echo "v12.7 strong serial DONE: $OUT_DIR"
