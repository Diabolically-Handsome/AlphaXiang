#!/bin/bash
# Queue v12.7 Light snapshot smokes on cuda:1 as snapshots appear.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
LIGHT_DIR="/home/laure/alphaxiang/training_runs/run_018a_v127_regret_light"
OUT_ROOT="/home/laure/alphaxiang/v127_snapshot_smoke"
SMOKE="$REPO/tools/_run_v127_snapshot_smoke_gpu1_serial.sh"
STEPS=(298000 300000 302000 304000)

mkdir -p "$OUT_ROOT"

light_running() {
    pgrep -af "xiangqi_train.py .*run_018a_v127_regret_light" >/dev/null 2>&1
}

for step in "${STEPS[@]}"; do
    ckpt="$LIGHT_DIR/snapshots/latest_step${step}.pt"
    out_dir="$OUT_ROOT/light_step${step}"
    done_marker="$out_dir/.done"
    failed_marker="$out_dir/.failed"
    if [ -f "$done_marker" ]; then
        echo "skip existing smoke: light_step${step}"
        continue
    fi
    while [ ! -f "$ckpt" ]; do
        if ! light_running; then
            echo "light training stopped before snapshot $step existed" >&2
            exit 1
        fi
        sleep 120
    done
    mkdir -p "$out_dir"
    echo "smoke start: step=$step ckpt=$ckpt out=$out_dir"
    if bash "$SMOKE" "$ckpt" "$out_dir" > "$out_dir/smoke_stdout.log" 2>&1; then
        touch "$done_marker"
        rm -f "$failed_marker"
        echo "smoke done: light_step${step}"
    else
        touch "$failed_marker"
        echo "smoke failed: light_step${step}" >&2
        exit 1
    fi
done

echo "v12.7 light snapshot queue DONE"
