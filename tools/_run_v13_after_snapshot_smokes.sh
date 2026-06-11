#!/bin/bash
# Wait for a v13 arm to produce selected snapshots, then run Pika smoke panels.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 ARM RUN_DIR" >&2
    echo "ARM should be dense or strategy; snapshots default to 100000..300000 every 25000." >&2
    exit 2
fi

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
ARM="$1"
RUN_DIR="$2"
OUT_ROOT="${V13_SMOKE_ROOT:-/home/laure/alphaxiang/v13_snapshot_smoke}"
SMOKE="$REPO/tools/_run_v13_snapshot_smoke.sh"
STEPS="${V13_SMOKE_STEPS:-100000 125000 150000 175000 200000 225000 250000 275000 300000}"
POLL_SECONDS="${V13_SMOKE_POLL_SECONDS:-180}"
MAX_WAIT_SECONDS="${V13_SMOKE_MAX_WAIT_SECONDS:-259200}"

if [ ! -f "$SMOKE" ]; then
    echo "missing smoke script: $SMOKE" >&2
    exit 1
fi

wait_for_checkpoint() {
    local ckpt="$1"
    local waited=0
    while [ ! -f "$ckpt" ]; do
        if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
            echo "timed out waiting for checkpoint: $ckpt" >&2
            return 1
        fi
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done
}

for STEP in $STEPS; do
    CKPT="$RUN_DIR/snapshots/latest_step${STEP}.pt"
    OUT_DIR="$OUT_ROOT/${ARM}_step${STEP}"
    DONE="$OUT_DIR/.done"
    FAILED="$OUT_DIR/.failed"
    if [ -f "$DONE" ]; then
        echo "skip existing v13 smoke: $OUT_DIR"
        continue
    fi
    echo "waiting for v13 snapshot: $CKPT"
    wait_for_checkpoint "$CKPT"
    mkdir -p "$OUT_DIR"
    echo "v13 smoke start: arm=$ARM step=$STEP ckpt=$CKPT out=$OUT_DIR"
    if bash "$SMOKE" "$CKPT" "$OUT_DIR" > "$OUT_DIR/smoke_stdout.log" 2>&1; then
        touch "$DONE"
        rm -f "$FAILED"
        echo "v13 smoke done: $OUT_DIR"
    else
        touch "$FAILED"
        echo "v13 smoke failed: $OUT_DIR" >&2
        exit 1
    fi
done
