#!/bin/bash
# Train a d5-focused surgical fallback arm with reset selfplay ingest state.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
DATA_ROOT="/home/laure/alphaxiang/v127_d5_surgical_data"
FAILURE_DIR="$DATA_ROOT/failure_d5"
ROOT_GUARD_DIR="$DATA_ROOT/root_guard_d5"
OUT_DIR="${V127_D5_SURGICAL_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018c_v127_d5_surgical}"
DEVICE="${V127_D5_SURGICAL_DEVICE:-cuda:0}"
MICRO_BATCH_SIZE="${V127_D5_SURGICAL_MICRO_BATCH_SIZE:-1024}"
GRAD_ACCUM_STEPS="${V127_D5_SURGICAL_GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${V127_D5_SURGICAL_MAX_STEPS:-302000}"

for run_dir in "$FAILURE_DIR" "$ROOT_GUARD_DIR"; do
    if [ ! -f "$run_dir/manifest.json" ]; then
        echo "missing manifest: $run_dir/manifest.json" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
cd "$REPO"

"$PY" xiangqi_train.py \
    --resume-path "$CKPT" \
    --reset-optimizer-on-resume \
    --reset-selfplay-ingest-state-on-resume \
    --selfplay-dirs "$FAILURE_DIR" "$ROOT_GUARD_DIR" \
    --output-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --foreground \
    --learning-rate 5e-5 \
    --warmup-steps 0 \
    --max-steps "$MAX_STEPS" \
    --lr-schedule-max-steps "$MAX_STEPS" \
    --snapshot-interval-steps 1000 \
    --save-interval-steps 1000 \
    --eval-interval-steps 1000 \
    --log-interval-steps 100 \
    --replay-buffer-size 4096 \
    --bootstrap-human-floor 0.10 \
    --samples-per-unit 64 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --teacher-q-loss-weight 0.25 \
    --teacher-q-temperature-cp 80 \
    --policy-oracle-alpha 0.40 \
    --value-loss-weight 2.0 \
    --seed 1270182 \
    2>&1 | tee "$OUT_DIR/train_stdout.log"

echo "v12.7 d5 surgical train DONE: $OUT_DIR"
