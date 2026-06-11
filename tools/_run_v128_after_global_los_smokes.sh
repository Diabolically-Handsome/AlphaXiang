#!/bin/bash
# Wait for v12.8G global+LOS warmup, then smoke selected snapshots.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
RUN_DIR="${V128_GLOBAL_LOS_DIR:-/home/laure/alphaxiang/training_runs/run_019h_v128_global_los_anchor}"
OUT_ROOT="${V128_SMOKE_ROOT:-/home/laure/alphaxiang/v128_snapshot_smoke}"
SMOKE="$REPO/tools/_run_v128_global_strategy_smoke.sh"
STEPS="${V128_GLOBAL_LOS_SMOKE_STEPS:-296500 297000}"
SMOKE_GAMES="${V128_SMOKE_GAMES:-12}"
D5_GAMES="${V128_D5_GAMES:-20}"

global_los_running() {
    pgrep -af "xiangqi_train.py .*run_019h_v128_global_los_anchor" >/dev/null 2>&1
}

while global_los_running; do
    sleep 120
done

for STEP in $STEPS; do
    CKPT="$RUN_DIR/snapshots/latest_step${STEP}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "missing v12.8G snapshot: $CKPT" >&2
        continue
    fi
    OUT_DIR="$OUT_ROOT/global_los_step${STEP}_quick"
    DONE="$OUT_DIR/.done"
    FAILED="$OUT_DIR/.failed"
    if [ -f "$DONE" ]; then
        echo "skip existing v12.8G smoke: $OUT_DIR"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "v12.8G global+LOS smoke start: ckpt=$CKPT out=$OUT_DIR"
    if V128_SMOKE_GAMES="$SMOKE_GAMES" V128_D5_GAMES="$D5_GAMES" bash "$SMOKE" "$CKPT" "$OUT_DIR" > "$OUT_DIR/smoke_stdout.log" 2>&1; then
        touch "$DONE"
        rm -f "$FAILED"
        echo "v12.8G global+LOS smoke done: $OUT_DIR"
    else
        touch "$FAILED"
        echo "v12.8G global+LOS smoke failed: $OUT_DIR" >&2
        exit 1
    fi
done
