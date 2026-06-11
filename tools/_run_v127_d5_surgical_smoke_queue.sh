#!/bin/bash
# Queue Pika d5-only smokes for the d5 surgical fallback arm.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
SURGICAL_DIR="${V127_D5_SURGICAL_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018c_v127_d5_surgical}"
OUT_ROOT="/home/laure/alphaxiang/v127_snapshot_smoke"
SMOKE="$REPO/tools/_run_v127_snapshot_d5_only_gpu1.sh"
STEPS=(297000 298000 299000 300000 301000 302000)

mkdir -p "$OUT_ROOT"

surgical_running() {
    pgrep -af "xiangqi_train.py .*run_018c_v127_d5_surgical" >/dev/null 2>&1
}

for step in "${STEPS[@]}"; do
    ckpt="$SURGICAL_DIR/snapshots/latest_step${step}.pt"
    out_dir="$OUT_ROOT/d5_surgical_step${step}_d5only"
    done_marker="$out_dir/.done"
    failed_marker="$out_dir/.failed"
    if [ -f "$done_marker" ]; then
        echo "skip existing d5-only smoke: d5_surgical_step${step}"
        continue
    fi
    while [ ! -f "$ckpt" ]; do
        if [ -f "$SURGICAL_DIR/.failed" ]; then
            echo "d5 surgical training failed before snapshot $step existed" >&2
            exit 1
        fi
        if [ -f "$SURGICAL_DIR/.done" ] && [ ! -f "$ckpt" ]; then
            echo "d5 surgical training completed but snapshot $step is missing" >&2
            exit 1
        fi
        if [ -d "$SURGICAL_DIR" ] && ! surgical_running; then
            echo "waiting for d5 surgical training snapshot $step"
        fi
        sleep 120
    done
    mkdir -p "$out_dir"
    echo "d5-only smoke start: step=$step ckpt=$ckpt out=$out_dir"
    if bash "$SMOKE" "$ckpt" "$out_dir" > "$out_dir/smoke_stdout.log" 2>&1; then
        touch "$done_marker"
        rm -f "$failed_marker"
        echo "d5-only smoke done: d5_surgical_step${step}"
    else
        touch "$failed_marker"
        echo "d5-only smoke failed: d5_surgical_step${step}" >&2
        exit 1
    fi
done

echo "v12.7 d5 surgical smoke queue DONE"
