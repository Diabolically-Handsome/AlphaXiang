#!/bin/bash
# After the d5 surgical train arm exits, smoke its snapshots, then resume Strong d5 smokes.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
SURGICAL_DIR="${V127_D5_SURGICAL_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018c_v127_d5_surgical}"
OUT_ROOT="/home/laure/alphaxiang/v127_snapshot_smoke"
D5_SMOKE="$REPO/tools/_run_v127_snapshot_d5_only_gpu1.sh"
STRONG_QUEUE="$REPO/tools/_run_v127_strong_d5_queue.sh"
SURGICAL_STEPS=(297000 298000)

surgical_running() {
    pgrep -af "xiangqi_train.py .*run_018c_v127_d5_surgical" >/dev/null 2>&1
}

while surgical_running; do
    sleep 120
done

mkdir -p "$OUT_ROOT"

for step in "${SURGICAL_STEPS[@]}"; do
    ckpt="$SURGICAL_DIR/snapshots/latest_step${step}.pt"
    out_dir="$OUT_ROOT/d5_surgical_step${step}_d5only"
    done_marker="$out_dir/.done"
    failed_marker="$out_dir/.failed"
    if [ ! -f "$ckpt" ]; then
        echo "skip missing surgical snapshot: $ckpt"
        continue
    fi
    if [ -f "$done_marker" ]; then
        echo "skip existing d5-only smoke: d5_surgical_step${step}"
        continue
    fi
    mkdir -p "$out_dir"
    echo "d5-only smoke start: step=$step ckpt=$ckpt out=$out_dir"
    if bash "$D5_SMOKE" "$ckpt" "$out_dir" > "$out_dir/smoke_stdout.log" 2>&1; then
        touch "$done_marker"
        rm -f "$failed_marker"
        echo "d5-only smoke done: d5_surgical_step${step}"
    else
        touch "$failed_marker"
        echo "d5-only smoke failed: d5_surgical_step${step}" >&2
        exit 1
    fi
done

if ! pgrep -af "tools/_run_v127_strong_d5_queue.sh" >/dev/null 2>&1; then
    echo "restarting Strong d5 queue"
    nohup bash "$STRONG_QUEUE" > "$OUT_ROOT/strong_d5_queue.log" 2>&1 < /dev/null &
else
    echo "Strong d5 queue already running"
fi

echo "v12.7 after d5 surgical smoke handoff DONE"
