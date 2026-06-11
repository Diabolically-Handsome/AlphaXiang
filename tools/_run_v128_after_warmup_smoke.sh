#!/bin/bash
# Wait for v12.8 relative-bias warmup, then smoke the final warmup snapshot.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
WARMUP_DIR="${V128_WARMUP_DIR:-/home/laure/alphaxiang/training_runs/run_019a_v128_relbias_warmup}"
OUT_ROOT="${V128_SMOKE_ROOT:-/home/laure/alphaxiang/v128_snapshot_smoke}"
SMOKE="$REPO/tools/_run_v128_snapshot_smoke.sh"
STEP="${V128_WARMUP_SMOKE_STEP:-298000}"

warmup_running() {
    pgrep -af "xiangqi_train.py .*run_019a_v128_relbias_warmup" >/dev/null 2>&1
}

while warmup_running; do
    sleep 120
done

CKPT="$WARMUP_DIR/snapshots/latest_step${STEP}.pt"
if [ ! -f "$CKPT" ]; then
    echo "missing warmup snapshot: $CKPT" >&2
    exit 1
fi

OUT_DIR="$OUT_ROOT/warmup_step${STEP}"
DONE="$OUT_DIR/.done"
FAILED="$OUT_DIR/.failed"
if [ -f "$DONE" ]; then
    echo "skip existing v12.8 warmup smoke: $OUT_DIR"
    exit 0
fi

mkdir -p "$OUT_DIR"
echo "v12.8 warmup smoke start: ckpt=$CKPT out=$OUT_DIR"
if bash "$SMOKE" "$CKPT" "$OUT_DIR" > "$OUT_DIR/smoke_stdout.log" 2>&1; then
    touch "$DONE"
    rm -f "$FAILED"
    echo "v12.8 warmup smoke done: $OUT_DIR"
else
    touch "$FAILED"
    echo "v12.8 warmup smoke failed: $OUT_DIR" >&2
    exit 1
fi
