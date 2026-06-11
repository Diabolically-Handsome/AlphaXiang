#!/bin/bash
# Start the v12.7 Strong arm only after the Light arm reaches its final snapshot.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
LIGHT_DIR="/home/laure/alphaxiang/training_runs/run_018a_v127_regret_light"
FINAL_CKPT="$LIGHT_DIR/snapshots/latest_step304000.pt"
STRONG_DIR="${V127_STRONG_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_018b_v127_regret_strong_serial}"
STRONG_SCRIPT="$REPO/tools/_run_v127_train_strong_serial.sh"
STARTED_MARKER="$STRONG_DIR/.started"
DONE_MARKER="$STRONG_DIR/.done"
FAILED_MARKER="$STRONG_DIR/.failed"

light_running() {
    pgrep -af "xiangqi_train.py .*run_018a_v127_regret_light" >/dev/null 2>&1
}

strong_running() {
    pgrep -af "xiangqi_train.py .*run_018b_v127_regret_strong_serial" >/dev/null 2>&1
}

while [ ! -f "$FINAL_CKPT" ]; do
    if ! light_running; then
        echo "light training is not running and final checkpoint is missing: $FINAL_CKPT" >&2
        exit 1
    fi
    sleep 180
done

while light_running; do
    echo "final Light snapshot exists; waiting for Light process to free cuda:0"
    sleep 60
done

if [ -f "$DONE_MARKER" ]; then
    echo "strong serial already completed: $STRONG_DIR"
    exit 0
fi
if [ -f "$STARTED_MARKER" ]; then
    if strong_running; then
        echo "strong serial already running: $STRONG_DIR"
        exit 0
    fi
    echo "removing stale strong start marker: $STARTED_MARKER"
    rm -f "$STARTED_MARKER"
fi

mkdir -p "$STRONG_DIR"
touch "$STARTED_MARKER"
rm -f "$FAILED_MARKER"

if bash "$STRONG_SCRIPT" > "$STRONG_DIR/strong_serial_stdout.log" 2>&1; then
    touch "$DONE_MARKER"
    echo "v12.7 strong serial completed: $STRONG_DIR"
else
    rm -f "$STARTED_MARKER"
    touch "$FAILED_MARKER"
    echo "v12.7 strong serial failed: $STRONG_DIR" >&2
    exit 1
fi
