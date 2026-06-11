#!/bin/bash
# v12.6 Day 1: positive control + ElephantEye depth ladder
# - Positive control: v12 vs Pikafish d=10 (does v12 hit a wall on in-dist strong opponent?)
# - Depth ladder: ElephantEye d=2/4/6/8 to find the non-saturated diagnostic depth
# All cells: sims=1600, 50 games, parallel-games=4
# Wall: ~75 min via 2 GPU queues

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v126_day1"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/pos_pika_d10" \
    "$OUT_BASE/eleeye_d2" \
    "$OUT_BASE/eleeye_d4" \
    "$OUT_BASE/eleeye_d6" \
    "$OUT_BASE/eleeye_d8"

run_arena() {
    local key="$1"
    local device="$2"
    local seed="$3"
    shift 3
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$V12" \
        --our-sims 1600 \
        --our-c-puct 1.25 \
        --our-q-weight 1.0 \
        --our-q-clip 1.0 \
        --our-temperature-move 0.1 \
        --games 50 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

# Queue cuda:0: positive control (slow) + ElephantEye d=8 (slow)
queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    # Positive control: v12 vs Pikafish d=10 (in-dist strong opponent test)
    run_arena "pos_pika_d10" cuda:0 33100 \
        --opp-engine pikafish --opp-depth 10
    # ElephantEye d=8 (probably hard, paired with positive control to balance load)
    run_arena "eleeye_d8" cuda:0 33108 \
        --opp-engine eleeye --opp-depth 8
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

# Queue cuda:1: ElephantEye d=2/4/6 (light to moderate)
queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_arena "eleeye_d2" cuda:1 33102 \
        --opp-engine eleeye --opp-depth 2
    run_arena "eleeye_d4" cuda:1 33104 \
        --opp-engine eleeye --opp-depth 4
    run_arena "eleeye_d6" cuda:1 33106 \
        --opp-engine eleeye --opp-depth 6
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "v12.6 Day 1 ALL DONE"
