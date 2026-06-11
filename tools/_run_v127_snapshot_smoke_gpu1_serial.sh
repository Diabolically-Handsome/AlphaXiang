#!/bin/bash
# Smoke-evaluate a v12.7 candidate checkpoint against Pika d3/d4/d5 on cuda:1.
# This serial variant is intended to run while cuda:0 is busy with finetuning.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 CHECKPOINT OUT_DIR" >&2
    exit 2
fi

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="$1"
OUT_BASE="$2"
DEVICE="${V127_SMOKE_DEVICE:-cuda:1}"
GAMES="${V127_SMOKE_GAMES:-50}"
PARALLEL_GAMES="${V127_SMOKE_PARALLEL_GAMES:-4}"

mkdir -p "$OUT_BASE/pika_d3" "$OUT_BASE/pika_d4" "$OUT_BASE/pika_d5"

run_external() {
    local key="$1"
    local depth="$2"
    local seed="$3"
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
        --games "$GAMES" \
        --parallel-games "$PARALLEL_GAMES" \
        --output-dir "$out_dir" \
        --device "$DEVICE" \
        --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|symbolic_guard_summary|loaded our model|value_source'
}

run_external "pika_d5" 5 127315
run_external "pika_d4" 4 127314
run_external "pika_d3" 3 127313

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                    "$OUT_BASE"/pika_d4/external_arena_*.json \
                    "$OUT_BASE"/pika_d5/external_arena_*.json \
    --json-out "$OUT_BASE/summary.json" \
    --markdown-out "$OUT_BASE/summary.md"

echo "v12.7 snapshot smoke DONE: $OUT_BASE"
