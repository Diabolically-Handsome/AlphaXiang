#!/bin/bash
# v12.7 root mate-in-1 guard revalidation.
# Runs the ship-candidate symbolic guard against Pika d3/d4/d5.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
OUT_BASE="/home/laure/alphaxiang/v127_guard_reval/root_guard_fixed"

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
        --games 200 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|symbolic_guard_summary|loaded our model|value_source'
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_external "pika_d3" 3 cuda:0 127003
    run_external "pika_d5" 5 cuda:0 127005
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_external "pika_d4" 4 cuda:1 127004
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"

STATUS=0
wait $PID0 || STATUS=$?
wait $PID1 || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "v12.7 root guard revalidation FAILED with status=$STATUS" >&2
    exit "$STATUS"
fi

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                    "$OUT_BASE"/pika_d4/external_arena_*.json \
                    "$OUT_BASE"/pika_d5/external_arena_*.json \
    --json-out "$OUT_BASE/summary.json" \
    --markdown-out "$OUT_BASE/summary.md"

echo "v12.7 root guard revalidation DONE: $OUT_BASE"
