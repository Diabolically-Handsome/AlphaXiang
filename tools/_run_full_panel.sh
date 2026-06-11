#!/bin/bash
# Full held-out panel evaluation: each of v4 / v5 / v6 vs each engine in the
# evaluation panel.  3 versions × 3 engines = 9 arena runs (CNN data is reused
# from earlier cyber dueling, since methodology + game count match).
#
# Each run: 50 games, sims=800, parallel-games=4 (with cross-game batcher).
# Per-run wall: ~5-7 min.  Two GPUs run two arenas in parallel.
# Total: ~25 min for all 9 runs.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/arena_runs/full_panel"

V4="/home/laure/alphaxiang/PEAK_step196000_v4_probe2_score63pct.pt"
V5="/home/laure/alphaxiang/PEAK_step204000_v5_probe1_score60pct_d2.pt"
V6="/home/laure/alphaxiang/PEAK_step210000_v6_probe2_score65pct_d3.pt"

GAMES=50
SIMS=800

mkdir -p "$OUT_BASE"

# One arena run.  Logs to file under $OUT_BASE/<key>/.
run_arena() {
    local key="$1"
    local ckpt="$2"
    local device="$3"
    local seed="$4"
    shift 4
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$ckpt" \
        --our-sims $SIMS \
        --games $GAMES \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        "$@" 2>&1 | tee "$out_dir/run.log" | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model|launched pikafish' || true
}

# Two parallel job queues, one per GPU.
# 5 jobs on cuda:0 (first 5), 4 jobs on cuda:1 (last 4).

queue_cuda0() {
    echo "[cuda:0] starting at $(date +%H:%M:%S)"
    run_arena "v4_pika_d1n15"  "$V4" cuda:0 1100 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v4_pika_d3"     "$V4" cuda:0 1101 \
        --opp-engine pikafish --opp-depth 3
    run_arena "v4_fairy_d3"    "$V4" cuda:0 1102 \
        --opp-engine fairy_sf --opp-depth 3
    run_arena "v5_pika_d1n15"  "$V5" cuda:0 1200 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v5_pika_d3"     "$V5" cuda:0 1201 \
        --opp-engine pikafish --opp-depth 3
    echo "[cuda:0] done at $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] starting at $(date +%H:%M:%S)"
    run_arena "v5_fairy_d3"    "$V5" cuda:1 1202 \
        --opp-engine fairy_sf --opp-depth 3
    run_arena "v6_pika_d1n15"  "$V6" cuda:1 1300 \
        --opp-engine pikafish --opp-depth 1 --opp-noise-ratio 0.15
    run_arena "v6_pika_d3"     "$V6" cuda:1 1301 \
        --opp-engine pikafish --opp-depth 3
    run_arena "v6_fairy_d3"    "$V6" cuda:1 1302 \
        --opp-engine fairy_sf --opp-depth 3
    echo "[cuda:1] done at $(date +%H:%M:%S)"
}

# Launch both queues, wait for both to finish.
queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!

echo "queues launched: cuda0_pid=$PID0 cuda1_pid=$PID1"
echo "waiting..."
wait $PID0 $PID1
echo "ALL DONE"
