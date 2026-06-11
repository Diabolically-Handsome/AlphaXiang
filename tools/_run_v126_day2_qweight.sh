#!/bin/bash
# v12.6 Day 2 Path A: q_weight grid vs Pika d=4
# Tests whether dampening the broken value head's influence helps
# All cells: sims=1600, c_puct=1.25, q_clip=1.0, noise=off, 50 games, parallel-games=4

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v126_day2_qweight"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/qw0.25" \
    "$OUT_BASE/qw0.50" \
    "$OUT_BASE/qw0.75" \
    "$OUT_BASE/qw1.50"

run_qw() {
    local key="$1"
    local device="$2"
    local seed="$3"
    local qw="$4"
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$V12" \
        --our-sims 1600 --our-c-puct 1.25 --our-q-clip 1.0 --our-temperature-move 0.1 \
        --our-q-weight "$qw" \
        --games 50 --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" --seed "$seed" \
        --opp-engine pikafish --opp-depth 4 \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_qw "qw0.25" cuda:0 34001 0.25
    run_qw "qw0.75" cuda:0 34003 0.75
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_qw "qw0.50" cuda:1 34002 0.50
    run_qw "qw1.50" cuda:1 34004 1.50
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "Day 2 q_weight grid DONE"
