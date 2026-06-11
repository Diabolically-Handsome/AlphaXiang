#!/bin/bash
# Wait for v12.8E anchored global-strategy warmup, then smoke selected snapshots.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
RUN_DIR="${V128_GLOBAL_STRATEGY_ANCHOR_DIR:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor}"
OUT_ROOT="${V128_SMOKE_ROOT:-/home/laure/alphaxiang/v128_snapshot_smoke}"
SMOKE="$REPO/tools/_run_v128_global_strategy_smoke.sh"
STEPS="${V128_GLOBAL_STRATEGY_ANCHOR_SMOKE_STEPS:-296500 297000}"

anchor_running() {
    pgrep -af "xiangqi_train.py .*run_019f_v128_global_strategy_anchor" >/dev/null 2>&1
}

while anchor_running; do
    sleep 120
done

for STEP in $STEPS; do
    CKPT="$RUN_DIR/snapshots/latest_step${STEP}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "missing anchored global-strategy snapshot: $CKPT" >&2
        continue
    fi
    OUT_DIR="$OUT_ROOT/global_strategy_anchor_step${STEP}"
    DONE="$OUT_DIR/.done"
    FAILED="$OUT_DIR/.failed"
    if [ -f "$DONE" ]; then
        echo "skip existing v12.8E smoke: $OUT_DIR"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "v12.8E anchored global-strategy smoke start: ckpt=$CKPT out=$OUT_DIR"
    if bash "$SMOKE" "$CKPT" "$OUT_DIR" > "$OUT_DIR/smoke_stdout.log" 2>&1; then
        touch "$DONE"
        rm -f "$FAILED"
        echo "v12.8E anchored global-strategy smoke done: $OUT_DIR"
    else
        touch "$FAILED"
        echo "v12.8E anchored global-strategy smoke failed: $OUT_DIR" >&2
        exit 1
    fi
done
