#!/bin/bash
# v7 vs full held-out panel: 4 engines × 50 games each, mirrors the v4/v5/v6 measurement.
# Two GPU queues, one per device — wall time ~30 min.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/arena_runs/v7_panel"
V7="/home/laure/alphaxiang/PEAK_step232500_v7_probe23_score72pct_d1.pt"

GAMES=50
SIMS=800

mkdir -p "$OUT_BASE"

run_arena() {
    local key="$1"
    local device="$2"
    local seed="$3"
    shift 3
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$V7" \
        --our-sims $SIMS \
        --games $GAMES \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model|launched pikafish' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_arena "v7_pika_d1n15" cuda:0 7100 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v7_pika_d3" cuda:0 7101 \
        --opp-engine pikafish --opp-depth 3
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_arena "v7_fairy_d3" cuda:1 7102 \
        --opp-engine fairy_sf --opp-depth 3
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

# CNN duel uses a different harness (transformer_vs_cnn_arena.py).  Run it serially
# AFTER cuda:0 queue finishes (since both want cuda:0).  Or put it on cuda:1 second.
queue_cnn_after_cuda1() {
    # wait for cuda:1 fairy job to finish
    wait $PID1
    echo "[cnn-on-cuda1] start $(date +%H:%M:%S)"
    cd "$REPO"
    "$PY" tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint "$V7" \
        --cnn-engine 'CNN/Chessv11_cpp_hist8_115_mps_fp16.py' \
        --cnn-weights 'CNN/best.pth' \
        --games 50 --sims 800 \
        --device cuda:1 \
        --output-dir "$OUT_BASE/v7_cnn" \
        --seed 7200 \
        2>&1 | tee "$OUT_BASE/v7_cnn/run.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate|loaded transformer|loaded CNN' || true
    echo "[cnn-on-cuda1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
queue_cnn_after_cuda1 > "$OUT_BASE/queue_cnn.log" 2>&1 &
PID2=$!

mkdir -p "$OUT_BASE/v7_cnn"
echo "queues: cuda0_pid=$PID0 cuda1_pid=$PID1 cnn_pid=$PID2"
wait $PID0 $PID1 $PID2
echo "ALL DONE"
