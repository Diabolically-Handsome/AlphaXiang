#!/bin/bash
# Launch v12.7 Light/Strong regret finetune arms from v12.6-micro.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
DATA_ROOT="/home/laure/alphaxiang/v127_regret_data"
FAILURE_DIR="$DATA_ROOT/failure_d4d5"
ROOT_GUARD_DIR="$DATA_ROOT/root_guard_events"
EXISTING_SLICE="/home/laure/alphaxiang/v126_day3_d4_slice"

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

cd "$REPO"

run_arm() {
    local name="$1"
    local device="$2"
    local out_dir="$3"
    local tq_weight="$4"
    local oracle_alpha="$5"
    local seed="$6"
    mkdir -p "$out_dir"
    "$PY" xiangqi_train.py \
        --resume-path "$CKPT" \
        --reset-optimizer-on-resume \
        --selfplay-dirs "$FAILURE_DIR" "$ROOT_GUARD_DIR" "$EXISTING_SLICE" \
        --output-dir "$out_dir" \
        --device "$device" \
        --foreground \
        --learning-rate 5e-5 \
        --warmup-steps 0 \
        --max-steps 304000 \
        --lr-schedule-max-steps 304000 \
        --snapshot-interval-steps 2000 \
        --save-interval-steps 2000 \
        --eval-interval-steps 2000 \
        --log-interval-steps 100 \
        --teacher-q-loss-weight "$tq_weight" \
        --teacher-q-temperature-cp 80 \
        --policy-oracle-alpha "$oracle_alpha" \
        --seed "$seed" \
        2>&1 | tee "$out_dir/train_stdout.log"
    echo "v12.7 arm DONE: $name -> $out_dir"
}

run_arm "light" cuda:0 /home/laure/alphaxiang/training_runs/run_018a_v127_regret_light 0.10 0.25 1270180 &
PID_LIGHT=$!
run_arm "strong" cuda:1 /home/laure/alphaxiang/training_runs/run_018b_v127_regret_strong 0.25 0.40 1270181 &
PID_STRONG=$!
echo "training arms launched: light=$PID_LIGHT strong=$PID_STRONG"

STATUS=0
wait $PID_LIGHT || STATUS=$?
wait $PID_STRONG || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "v12.7 training arms FAILED with status=$STATUS" >&2
    exit "$STATUS"
fi

echo "v12.7 training arms DONE"
