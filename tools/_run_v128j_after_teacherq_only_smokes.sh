#!/bin/bash
# Wait for v12.8J teacher-Q-only finetune, then smoke selected snapshots.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
RUN_DIR="${V128J_RUN_DIR:-/home/laure/alphaxiang/training_runs/run_019l_v128j_teacherq_only_refute}"
OUT_ROOT="${V128J_SMOKE_ROOT:-/home/laure/alphaxiang/v128_snapshot_smoke}"
SMOKE="$REPO/tools/_run_v128_global_strategy_smoke.sh"
STEPS="${V128J_SMOKE_STEPS:-297250 297500}"
SMOKE_GAMES="${V128J_SMOKE_GAMES:-20}"
D5_GAMES="${V128J_D5_GAMES:-50}"

teacherq_only_running() {
    pgrep -af "xiangqi_train.py .*$(basename "$RUN_DIR")" >/dev/null 2>&1
}

while teacherq_only_running; do
    sleep 120
done

for STEP in $STEPS; do
    CKPT="$RUN_DIR/snapshots/latest_step${STEP}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "missing v12.8J snapshot: $CKPT" >&2
        continue
    fi
    OUT_DIR="$OUT_ROOT/v128j_teacherq_only_step${STEP}"
    DONE="$OUT_DIR/.done"
    FAILED="$OUT_DIR/.failed"
    if [ -f "$DONE" ]; then
        echo "skip existing v12.8J smoke: $OUT_DIR"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "v12.8J teacher-Q-only smoke start: ckpt=$CKPT out=$OUT_DIR"
    if V128_SMOKE_GAMES="$SMOKE_GAMES" V128_D5_GAMES="$D5_GAMES" bash "$SMOKE" "$CKPT" "$OUT_DIR" > "$OUT_DIR/smoke_stdout.log" 2>&1; then
        touch "$DONE"
        rm -f "$FAILED"
        echo "v12.8J teacher-Q-only smoke done: $OUT_DIR"
    else
        touch "$FAILED"
        echo "v12.8J teacher-Q-only smoke failed: $OUT_DIR" >&2
        exit 1
    fi
done
