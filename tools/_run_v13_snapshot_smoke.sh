#!/bin/bash
# v13 200M snapshot smoke: Pika d3/d4/d5, 20 games by default.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 CHECKPOINT OUT_DIR" >&2
    exit 2
fi

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="$1"
OUT_BASE="$2"
GAMES="${V13_SMOKE_GAMES:-20}"
PARALLEL_GAMES="${V13_PARALLEL_GAMES:-4}"
D3_DEVICE="${V13_D3_DEVICE:-cuda:0}"
D4_DEVICE="${V13_D4_DEVICE:-cuda:1}"
D5_DEVICE="${V13_D5_DEVICE:-cuda:0}"

if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint: $CKPT" >&2
    exit 1
fi

mkdir -p "$OUT_BASE/pika_d3" "$OUT_BASE/pika_d4" "$OUT_BASE/pika_d5"

run_external() {
    local key="$1"
    local depth="$2"
    local device="$3"
    local seed="$4"
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$CKPT" \
        --our-sims 1600 \
        --our-c-puct 1.25 \
        --our-q-weight 1.0 \
        --our-q-clip 1.0 \
        --our-value-source scalar \
        --our-temperature-move 0.1 \
        --games "$GAMES" \
        --parallel-games "$PARALLEL_GAMES" \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model|value_source'
}

run_external "pika_d3" 3 "$D3_DEVICE" 130003 &
PID3=$!
run_external "pika_d4" 4 "$D4_DEVICE" 130004 &
PID4=$!
STATUS=0
wait "$PID3" || STATUS=$?
wait "$PID4" || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "v13 d3/d4 smoke failed with status=$STATUS" >&2
    exit "$STATUS"
fi

run_external "pika_d5" 5 "$D5_DEVICE" 130005

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                    "$OUT_BASE"/pika_d4/external_arena_*.json \
                    "$OUT_BASE"/pika_d5/external_arena_*.json \
    --json-out "$OUT_BASE/summary.json" \
    --markdown-out "$OUT_BASE/summary.md"

echo "v13 snapshot smoke DONE: $OUT_BASE"
