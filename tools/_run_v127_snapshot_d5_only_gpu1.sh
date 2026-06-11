#!/bin/bash
# Smoke-evaluate only the Pika d5 bottleneck cell for a v12.7 checkpoint.

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

mkdir -p "$OUT_BASE/pika_d5"

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
    --output-dir "$OUT_BASE/pika_d5" \
    --device "$DEVICE" \
    --seed 127315 \
    --opp-engine pikafish --opp-depth 5 \
    2>&1 | tee "$OUT_BASE/pika_d5/run.log" \
    | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|symbolic_guard_summary|loaded our model|value_source'

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d5/external_arena_*.json \
    --json-out "$OUT_BASE/summary_d5_only.json" \
    --markdown-out "$OUT_BASE/summary_d5_only.md"

echo "v12.7 d5-only snapshot smoke DONE: $OUT_BASE"
