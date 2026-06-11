#!/bin/bash
# v12.6-micro Path D evaluation: arena tests on the trained snapshot
# Test step296000 vs Pika d=3 (cuda:0) and d=4 (cuda:1) in parallel
# Wall: ~16 min parallel

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
OUT_BASE="/home/laure/alphaxiang/v126_micro_eval"

mkdir -p "$OUT_BASE/d3" "$OUT_BASE/d4"

run_arena() {
    local key="$1"
    local depth="$2"
    local device="$3"
    local seed="$4"
    local out_dir="$OUT_BASE/$key"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$CKPT" \
        --our-sims 1600 --our-c-puct 1.25 --our-q-weight 1.0 --our-q-clip 1.0 \
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
    run_arena "d3" 3 cuda:0 51003
    echo "[cuda:0] done $(date +%H:%M:%S)"
}
queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_arena "d4" 4 cuda:1 51004
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
wait $PID0 $PID1
echo "v12.6-micro arena eval DONE"
