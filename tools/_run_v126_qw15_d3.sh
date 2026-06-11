#!/bin/bash
# Quick verification: q_weight=1.5 vs Pika d=3
# Tests whether q_w=1.5 is the "safe middle" that preserves d=3 strength
# while modestly helping d=4+
set -e
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"
OUT_DIR="/home/laure/alphaxiang/v126_qw15_d3"
mkdir -p "$OUT_DIR"
cd "$REPO"
"$PY" tools/external_arena.py \
    --checkpoint "$V12" \
    --our-sims 1600 --our-c-puct 1.25 --our-q-clip 1.0 --our-temperature-move 0.1 \
    --our-q-weight 1.5 \
    --games 50 --parallel-games 4 \
    --output-dir "$OUT_DIR" \
    --device cuda:0 --seed 36003 \
    --opp-engine pikafish --opp-depth 3 \
    2>&1 | tee "$OUT_DIR/run.log" \
    | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
echo "qw=1.5 vs d=3 DONE"
