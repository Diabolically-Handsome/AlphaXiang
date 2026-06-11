#!/bin/bash
# v12.6 Day 1 (revised): Pikafish depth ladder — replaces stalled ElephantEye ladder
# d=4,5,6,7 should bracket the 25-75% v12 diagnostic zone
# All cells: sims=1600, 50 games, parallel-games=4
# Wall: ~60 min via 2 GPU queues

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v126_day1"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/pika_d4" \
    "$OUT_BASE/pika_d5" \
    "$OUT_BASE/pika_d6" \
    "$OUT_BASE/pika_d7"

run_pika() {
    local key="$1"
    local device="$2"
    local seed="$3"
    local depth="$4"
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$V12" \
        --our-sims 1600 \
        --our-c-puct 1.25 --our-q-weight 1.0 --our-q-clip 1.0 \
        --our-temperature-move 0.1 \
        --games 50 --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_pika "pika_d4" cuda:0 33204 4
    run_pika "pika_d6" cuda:0 33206 6
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_pika "pika_d5" cuda:1 33205 5
    run_pika "pika_d7" cuda:1 33207 7
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_pika_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_pika_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "Pika depth ladder DONE"
