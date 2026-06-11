#!/bin/bash
# v12.5 Phase C-1: search knob grid vs ElephantEye d=10
# 4 orthogonal cells testing whether any single search knob can crack 0/50.
# All cells: sims=1600, 50 games, eleeye d=10. Vary one knob from default each.

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_BASE="/home/laure/alphaxiang/v125_eleeye_grid"
V12="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

mkdir -p "$OUT_BASE" \
    "$OUT_BASE/A_qw0.5" \
    "$OUT_BASE/B_qw2.0" \
    "$OUT_BASE/C_cpuct0.8" \
    "$OUT_BASE/D_noise_on"

run_eleeye() {
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
        --our-q-clip 1.0 \
        --our-temperature-move 0.1 \
        --games 50 \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        --opp-engine eleeye --opp-depth 10 \
        "$@" 2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
}

queue_cuda0() {
    echo "[cuda:0] start $(date +%H:%M:%S)"
    # Cell A: q_weight=0.5 (trust policy more)
    run_eleeye "A_qw0.5" cuda:0 32100 \
        --our-c-puct 1.25 --our-q-weight 0.5
    # Cell C: c_puct=0.8 (less exploration, commit to top moves)
    run_eleeye "C_cpuct0.8" cuda:0 32200 \
        --our-c-puct 0.8 --our-q-weight 1.0
    echo "[cuda:0] done $(date +%H:%M:%S)"
}

queue_cuda1() {
    echo "[cuda:1] start $(date +%H:%M:%S)"
    # Cell B: q_weight=2.0 (trust value more)
    run_eleeye "B_qw2.0" cuda:1 32300 \
        --our-c-puct 1.25 --our-q-weight 2.0
    # Cell D: root_noise=on (break tactical preparation, force diversity)
    run_eleeye "D_noise_on" cuda:1 32400 \
        --our-c-puct 1.25 --our-q-weight 1.0 --our-add-root-noise
    echo "[cuda:1] done $(date +%H:%M:%S)"
}

queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
PID0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
PID1=$!
echo "queues launched: pid0=$PID0 pid1=$PID1"
wait $PID0 $PID1
echo "C-1 ALL DONE"
