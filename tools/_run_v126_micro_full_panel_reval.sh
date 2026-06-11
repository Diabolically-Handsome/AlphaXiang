#!/bin/bash
# v12.6-micro full panel re-evaluation.
# This intentionally writes to a fresh output directory and does not touch the
# original v12.6-micro d3/d4 results under /home/laure/alphaxiang/v126_micro_eval.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
OUT_BASE="/home/laure/alphaxiang/v126_micro_full_panel_reval"

mkdir -p \
    "$OUT_BASE/pika_d1n15" \
    "$OUT_BASE/pika_d5" \
    "$OUT_BASE/fairy_d3" \
    "$OUT_BASE/cnn_best"

run_external() {
    local key="$1"
    local device="$2"
    local seed="$3"
    shift 3
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
        --games 50 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model'
}

run_cnn() {
    local out_dir="$OUT_BASE/cnn_best"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint "$CKPT" \
        --cnn-engine CNN/Chessv11_cpp_hist8_115_mps_fp16.py \
        --cnn-weights CNN/best.pth \
        --games 50 \
        --sims 1600 \
        --c-puct 1.25 \
        --temperature-move 0.1 \
        --device cuda:1 \
        --output-dir "$out_dir" \
        --seed 62211 2>&1 | tee "$out_dir/run.log"
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_external "pika_d1n15" cuda:0 62001 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_external "fairy_d3" cuda:0 62103 \
        --opp-engine fairy_sf --opp-depth 3
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_external "pika_d5" cuda:1 62005 \
        --opp-engine pikafish --opp-depth 5
    run_cnn
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
    echo "v12.6-micro full panel re-eval FAILED with status=$STATUS" >&2
    exit "$STATUS"
fi

"$PY" tools/summarize_panel_results.py \
    --external-json \
        /home/laure/alphaxiang/v126_micro_eval/d3/external_arena_*.json \
        /home/laure/alphaxiang/v126_micro_eval/d4/external_arena_*.json \
        "$OUT_BASE"/pika_d1n15/external_arena_*.json \
        "$OUT_BASE"/pika_d5/external_arena_*.json \
        "$OUT_BASE"/fairy_d3/external_arena_*.json \
    --cnn-json "$OUT_BASE"/cnn_best/tournament_*.json \
    --json-out "$OUT_BASE/summary.json" \
    --markdown-out "$OUT_BASE/summary.md"

echo "v12.6-micro full panel re-eval DONE: $OUT_BASE"
