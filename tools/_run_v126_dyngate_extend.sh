#!/bin/bash
# v12.6-micro Path C extension: dyngate aggr (1.0/1.5/2.0) on Fairy + Pika d=5
# To complete the 4-axis check (d=3, d=4, d=5, Fairy)
# All sims=1600, 50 games, parallel-games=4, cuda:1
# Wall: ~30 min sequential

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"
OUT_BASE="/home/laure/alphaxiang/v126_dyngate"

mkdir -p "$OUT_BASE/B_aggr_d5" "$OUT_BASE/B_aggr_fairy_d3"

run_dyngate() {
    local key="$1"
    local seed="$2"
    shift 2
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena_dyngate.py \
        --checkpoint "$V12" \
        --our-sims 1600 --our-c-puct 1.25 --our-q-clip 1.0 --our-temperature-move 0.1 \
        --our-q-weight-open 1.0 --our-q-weight-mid 1.5 --our-q-weight-end 2.0 \
        --our-q-weight-mid-ply 30 --our-q-weight-end-ply 80 \
        --games 50 --parallel-games 4 \
        --output-dir "$out_dir" \
        --device cuda:1 --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

echo "[dyngate-extend] start $(date +%H:%M:%S)"

# B_aggr_d5: dyngate aggr vs Pikafish d=5
run_dyngate "B_aggr_d5" 42005 \
    --opp-engine pikafish --opp-depth 5

# B_aggr_fairy_d3: dyngate aggr vs Fairy-SF d=3
run_dyngate "B_aggr_fairy_d3" 42103 \
    --opp-engine fairy_sf --opp-depth 3

echo "[dyngate-extend] done $(date +%H:%M:%S)"
