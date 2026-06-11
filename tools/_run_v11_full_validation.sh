#!/bin/bash
# v11 full validation suite:
# - Standard 4-engine panel (Pikafish d=1+n0.15 / Pikafish d=3 / Fairy-SF d=3 / CNN best)
# - ElephantEye d=10 anchor (public-ladder calibration)
# - v11 vs v10 head-to-head (50 games)
# Two parallel GPU queues. Wall time ~75-90 min (CNN dominates cuda:1 queue).
#
# Bug-fix vs _run_v10_panel.sh: CNN queue runs INSIDE the same shell function as fairy
# (sequential in cuda:1 queue), so no cross-shell `wait $PID1` issue.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/arena_runs/v11_panel"
V11="/home/laure/alphaxiang/PEAK_step270000_v11_probe2_score90pct_d1.pt"
V10="/home/laure/alphaxiang/PEAK_step255000_v10_probe3_score77pct_d1.pt"

GAMES=50
SIMS=800

mkdir -p "$OUT_BASE" "$OUT_BASE/v11_cnn" "$OUT_BASE/v11_eleeye" "$OUT_BASE/v11_vs_v10"

run_arena() {
    local key="$1"
    local device="$2"
    local seed="$3"
    shift 3
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$V11" \
        --our-sims $SIMS \
        --games $GAMES \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    run_arena "v11_pika_d1n15" cuda:0 11100 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v11_pika_d3" cuda:0 11101 \
        --opp-engine pikafish --opp-depth 3
    run_arena "v11_eleeye" cuda:0 11102 \
        --opp-engine eleeye --opp-depth 10
    # v11 vs v10 head-to-head
    cd "$REPO"
    "$PY" tools/transformer_vs_transformer_arena.py \
        --a-checkpoint "$V11" \
        --b-checkpoint "$V10" \
        --games 50 --sims $SIMS \
        --device cuda:0 \
        --output-dir "$OUT_BASE/v11_vs_v10" \
        --seed 11200 \
        2>&1 | tee "$OUT_BASE/v11_vs_v10/duel.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_arena "v11_fairy_d3" cuda:1 11103 \
        --opp-engine fairy_sf --opp-depth 3
    # CNN match runs serially after fairy in same shell — no cross-shell wait
    cd "$REPO"
    "$PY" tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint "$V11" \
        --cnn-engine 'CNN/Chessv11_cpp_hist8_115_mps_fp16.py' \
        --cnn-weights 'CNN/best.pth' \
        --games 50 --sims $SIMS \
        --device cuda:1 \
        --output-dir "$OUT_BASE/v11_cnn" \
        --seed 11201 \
        2>&1 | tee "$OUT_BASE/v11_cnn/run.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!

echo "queues launched: cuda0_pid=$PID0 cuda1_pid=$PID1"
wait $PID0 $PID1
echo "ALL DONE"
