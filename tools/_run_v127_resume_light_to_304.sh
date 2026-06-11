#!/bin/bash
# Resume the interrupted v12.7 Light arm from its latest checkpoint to 304k.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
DATA_ROOT="/home/laure/alphaxiang/v127_regret_data"
FAILURE_DIR="$DATA_ROOT/failure_d4d5"
ROOT_GUARD_DIR="$DATA_ROOT/root_guard_events"
EXISTING_SLICE="/home/laure/alphaxiang/v126_day3_d4_slice"
OUT_DIR="/home/laure/alphaxiang/training_runs/run_018a_v127_regret_light"
CKPT="$OUT_DIR/latest.pt"
DEVICE="${V127_LIGHT_RESUME_DEVICE:-cuda:0}"

if [ ! -f "$CKPT" ]; then
    echo "missing Light resume checkpoint: $CKPT" >&2
    exit 1
fi
for run_dir in "$FAILURE_DIR" "$ROOT_GUARD_DIR" "$EXISTING_SLICE"; do
    if [ ! -f "$run_dir/manifest.json" ]; then
        echo "missing manifest: $run_dir/manifest.json" >&2
        exit 1
    fi
done

cd "$REPO"
"$PY" xiangqi_train.py \
    --resume-path "$CKPT" \
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
    --teacher-q-loss-weight 0.10 \
    --teacher-q-temperature-cp 80 \
    --policy-oracle-alpha 0.25 \
    --seed 1270180 \
    2>&1 | tee -a "$OUT_DIR/train_stdout.log"

echo "v12.7 Light resume DONE: $OUT_DIR"
