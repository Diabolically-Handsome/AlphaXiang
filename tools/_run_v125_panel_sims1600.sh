#!/bin/bash
# v12.5 Stage B-1 + B-2: full panel re-evaluation at sims=1600
# Pika d=3 already covered by diagnostic grid (200 games, 48.75%) — skip here.
# 4 cells total, 2 parallel queues (cuda:0 and cuda:1).
# Wall: ~60-90 min.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v125_panel_sims1600"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"
SIMS=1600

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/v12_pika_d1n15_s1600" \
    "$OUT_BASE/v12_fairy_d3_s1600" \
    "$OUT_BASE/v12_cnn_s1600" \
    "$OUT_BASE/v12_eleeye_d10_s1600"

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
        --our-sims $SIMS \
        --games 50 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    # B-1 cell 1: Pikafish d=1 + n=0.15 (saturated opponent, sanity check)
    run_arena "v12_pika_d1n15_s1600" cuda:0 31100 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    # B-2: ElephantEye d=10 (the public-Elo gap test)
    run_arena "v12_eleeye_d10_s1600" cuda:0 31200 \
        --opp-engine eleeye --opp-depth 10
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    # B-1 cell 2: Fairy-SF d=3 (alternative-engine; expect big jump)
    run_arena "v12_fairy_d3_s1600" cuda:1 31300 \
        --opp-engine fairy_sf --opp-depth 3
    # B-1 cell 3: CNN best (test if regression at sims=800 was a search artifact)
    cd "$REPO"
    "$PY" tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint "$V12" \
        --cnn-engine 'CNN/Chessv11_cpp_hist8_115_mps_fp16.py' \
        --cnn-weights 'CNN/best.pth' \
        --games 50 --sims $SIMS --device cuda:1 \
        --output-dir "$OUT_BASE/v12_cnn_s1600" --seed 31400 \
        2>&1 | tee "$OUT_BASE/v12_cnn_s1600/run.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "B-1 + B-2 ALL DONE"
