#!/bin/bash
# Queue Pika d5-only smokes for v12.7 Strong snapshots.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
STRONG_DIR="${V127_STRONG_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018b_v127_regret_strong_serial}"
OUT_ROOT="/home/laure/alphaxiang/v127_snapshot_smoke"
SMOKE="$REPO/tools/_run_v127_snapshot_d5_only_gpu1.sh"
STEPS=(298000 300000 302000 304000)

mkdir -p "$OUT_ROOT"

strong_running() {
    pgrep -af "xiangqi_train.py .*run_018b_v127_regret_strong_serial" >/dev/null 2>&1
}

for step in "${STEPS[@]}"; do
    ckpt="$STRONG_DIR/snapshots/latest_step${step}.pt"
    out_dir="$OUT_ROOT/strong_step${step}_d5only"
    done_marker="$out_dir/.done"
    failed_marker="$out_dir/.failed"
    if [ -f "$done_marker" ]; then
        echo "skip existing d5-only smoke: strong_step${step}"
        continue
    fi
    while [ ! -f "$ckpt" ]; do
        if [ -f "$STRONG_DIR/.failed" ]; then
            echo "strong training failed before snapshot $step existed" >&2
            exit 1
        fi
        if [ -f "$STRONG_DIR/.done" ] && [ ! -f "$ckpt" ]; then
            echo "strong training completed but snapshot $step is missing" >&2
            exit 1
        fi
        if [ -d "$STRONG_DIR" ] && ! strong_running && [ ! -f "$STRONG_DIR/.started" ]; then
            echo "waiting for strong training to start"
        fi
        sleep 120
    done
    mkdir -p "$out_dir"
    echo "d5-only smoke start: step=$step ckpt=$ckpt out=$out_dir"
    if bash "$SMOKE" "$ckpt" "$out_dir" > "$out_dir/smoke_stdout.log" 2>&1; then
        touch "$done_marker"
        rm -f "$failed_marker"
        echo "d5-only smoke done: strong_step${step}"
    else
        touch "$failed_marker"
        echo "d5-only smoke failed: strong_step${step}" >&2
        exit 1
    fi
done

echo "v12.7 Strong d5-only queue DONE"
