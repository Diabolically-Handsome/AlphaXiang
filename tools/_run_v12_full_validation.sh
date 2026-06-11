#!/bin/bash
# v12 full validation suite, mirroring v11's:
# - Standard 4-engine panel (Pikafish d=1+n0.15 / Pikafish d=3 / Fairy-SF d=3 / CNN best)
# - v12 vs v11 head-to-head (50 games)
# - v12 vs v10 head-to-head (50 games)
# - v12 vs v7 head-to-head (50 games)
# (Skipping ElephantEye anchor since v11's was 0-44, no new info expected)
# Two parallel GPU queues. Wall time ~120 min.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/arena_runs/v12_panel"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"
V11="/home/laure/alphaxiang/PEAK_step270000_v11_probe2_score90pct_d1.pt"
V10="/home/laure/alphaxiang/PEAK_step255000_v10_probe3_score77pct_d1.pt"
V7="/home/laure/alphaxiang/PEAK_step232500_v7_probe23_score72pct_d1.pt"

GAMES=50
SIMS=800

mkdir -p "$OUT_BASE" "$OUT_BASE/v12_cnn" "$OUT_BASE/v12_vs_v11" "$OUT_BASE/v12_vs_v10" "$OUT_BASE/v12_vs_v7"

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
    run_arena "v12_pika_d1n15" cuda:0 12100 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v12_pika_d3" cuda:0 12101 \
        --opp-engine pikafish --opp-depth 3
    # v12 vs v11 head-to-head
    cd "$REPO"
    "$PY" tools/transformer_vs_transformer_arena.py \
        --a-checkpoint "$V12" --a-label v12 \
        --b-checkpoint "$V11" --b-label v11 \
        --games 50 --sims $SIMS --device cuda:0 \
        --output-dir "$OUT_BASE/v12_vs_v11" --seed 12200 \
        2>&1 | tee "$OUT_BASE/v12_vs_v11/duel.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    # v12 vs v10 head-to-head
    cd "$REPO"
    "$PY" tools/transformer_vs_transformer_arena.py \
        --a-checkpoint "$V12" --a-label v12 \
        --b-checkpoint "$V10" --b-label v10 \
        --games 50 --sims $SIMS --device cuda:0 \
        --output-dir "$OUT_BASE/v12_vs_v10" --seed 12201 \
        2>&1 | tee "$OUT_BASE/v12_vs_v10/duel.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    # v12 vs v7 head-to-head (skip-generation regression check)
    cd "$REPO"
    "$PY" tools/transformer_vs_transformer_arena.py \
        --a-checkpoint "$V12" --a-label v12 \
        --b-checkpoint "$V7" --b-label v7 \
        --games 50 --sims $SIMS --device cuda:0 \
        --output-dir "$OUT_BASE/v12_vs_v7" --seed 12202 \
        2>&1 | tee "$OUT_BASE/v12_vs_v7/duel.log" \
        | grep -E '^game [0-9]|^FINAL:|score_rate' || true
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    run_arena "v12_fairy_d3" cuda:1 12103 \
        --opp-engine fairy_sf --opp-depth 3
    cd "$REPO"
    "$PY" tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint "$V12" \
        --cnn-engine 'CNN/Chessv11_cpp_hist8_115_mps_fp16.py' \
        --cnn-weights 'CNN/best.pth' \
        --games 50 --sims $SIMS --device cuda:1 \
        --output-dir "$OUT_BASE/v12_cnn" --seed 12201 \
        2>&1 | tee "$OUT_BASE/v12_cnn/run.log" \
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
