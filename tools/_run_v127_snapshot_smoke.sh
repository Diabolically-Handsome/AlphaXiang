#!/bin/bash
# Smoke-evaluate a v12.7 candidate checkpoint against Pika d3/d4/d5.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 CHECKPOINT OUT_DIR" >&2
    exit 2
fi

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="$1"
OUT_BASE="$2"

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
        --our-root-mate1-blunder-guard \
        --our-temperature-move 0.1 \
        --games 50 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|symbolic_guard_summary|loaded our model|value_source'
}

run_external "pika_d3" 3 cuda:0 127303 &
PID3=$!
run_external "pika_d4" 4 cuda:1 127304 &
PID4=$!
STATUS=0
wait $PID3 || STATUS=$?
wait $PID4 || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "v12.7 d3/d4 smoke FAILED with status=$STATUS" >&2
    exit "$STATUS"
fi
run_external "pika_d5" 5 cuda:0 127305

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                    "$OUT_BASE"/pika_d4/external_arena_*.json \
                    "$OUT_BASE"/pika_d5/external_arena_*.json \
    --json-out "$OUT_BASE/summary.json" \
    --markdown-out "$OUT_BASE/summary.md"

echo "v12.7 snapshot smoke DONE: $OUT_BASE"
