#!/bin/bash
# v12.6-micro Path C: dyngate q_weight tests
# 4 cells: 2 configs × 2 opponents (Pika d=3, d=4)
# Config A "conservative": open=1.0/mid=1.2/end=1.5
# Config B "aggressive":   open=1.0/mid=1.5/end=2.0
# All sims=1600, 50 games, parallel-games=4, cuda:1
# Wall: ~64 min sequential

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"
OUT_BASE="/home/laure/alphaxiang/v126_dyngate"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/A_cons_d3" "$OUT_BASE/A_cons_d4" \
    "$OUT_BASE/B_aggr_d3" "$OUT_BASE/B_aggr_d4"

run_dyngate() {
    local key="$1"
    local depth="$2"
    local seed="$3"
    local open_qw="$4"
    local mid_qw="$5"
    local end_qw="$6"
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena_dyngate.py \
        --checkpoint "$V12" \
        --our-sims 1600 --our-c-puct 1.25 --our-q-clip 1.0 --our-temperature-move 0.1 \
        --our-q-weight-open "$open_qw" --our-q-weight-mid "$mid_qw" --our-q-weight-end "$end_qw" \
        --our-q-weight-mid-ply 30 --our-q-weight-end-ply 80 \
        --games 50 --parallel-games 4 \
        --output-dir "$out_dir" \
        --device cuda:1 --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

echo "[dyngate] start $(date +%H:%M:%S)"

# Config A "conservative" 1.0/1.2/1.5
run_dyngate "A_cons_d3" 3 41003 1.0 1.2 1.5
run_dyngate "A_cons_d4" 4 41004 1.0 1.2 1.5

# Config B "aggressive" 1.0/1.5/2.0
run_dyngate "B_aggr_d3" 3 42003 1.0 1.5 2.0
run_dyngate "B_aggr_d4" 4 42004 1.0 1.5 2.0

echo "[dyngate] done $(date +%H:%M:%S)"
