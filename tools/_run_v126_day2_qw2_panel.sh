#!/bin/bash
# v12.6 Day 2 extension: panel re-eval at q_weight=2.0
# Verify whether q_w=2.0 generalizes (helps d=4 +13pp; does it hurt d=3 / Fairy / CNN?)
# 4 cells: Pika d=3, Pika d=5, Fairy-SF d=3, CNN
# All sims=1600, 50 games, parallel-games=4
# Wall: ~50 min via 2 GPU queues

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v126_day2_qw2_panel"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/qw2_pika_d3" \
    "$OUT_BASE/qw2_pika_d4_replicate" \
    "$OUT_BASE/qw2_pika_d5" \
    "$OUT_BASE/qw2_fairy_d3"

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
        --our-sims 1600 --our-c-puct 1.25 --our-q-clip 1.0 --our-temperature-move 0.1 \
        --our-q-weight 2.0 \
        --games 50 --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    # Pika d=3 (currently 49.25% — does q_w=2.0 hurt this?)
    run_arena "qw2_pika_d3" cuda:0 35001 \
        --opp-engine pikafish --opp-depth 3
    # Pika d=5 (currently 8% — does q_w=2.0 help this?)
    run_arena "qw2_pika_d5" cuda:0 35005 \
        --opp-engine pikafish --opp-depth 5
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    # Pika d=4 replicate (different seed; verify 33% holds, not lucky)
    run_arena "qw2_pika_d4_replicate" cuda:1 35004 \
        --opp-engine pikafish --opp-depth 4
    # Fairy-SF d=3 (currently 82% — saturate or move?)
    run_arena "qw2_fairy_d3" cuda:1 35003 \
        --opp-engine fairy_sf --opp-depth 3
    # NOTE: CNN intentionally skipped — transformer_vs_cnn_arena.py has q_weight=1.0
    # hardcoded; need to add CLI flag to test q_w=2.0 there. Defer to user.
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "Day 2 q_w=2.0 panel re-eval DONE"
