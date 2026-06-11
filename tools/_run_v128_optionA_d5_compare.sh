#!/usr/bin/env bash
# Option A: 4-way d5 arena comparison at identical protocol (black-side, 6400 sims).
# Answers: (1) does V12.8 retrain beat Probe B / v12? (2) V12.8 vs V12.6 d5 win rate, clean same-protocol.
# Protocol mirrors tools/_run_v128_d20_root_scaleup.sh run_collect_depth exactly.
set -uo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"

OUT_BASE="/home/laure/alphaxiang/v128_optionA_d5_compare"
OPENING_SUITE="arena_openings/human_val_opening_suite_v1.json"
SEED=2026060101
DEPTH=5

V128_RETRAIN="/home/laure/alphaxiang/training_runs/run_025_v128_d20_stage2_20k_smoke/full_model_200step_more_conservative_pure_root/latest.pt"
PROBE_B="/home/laure/alphaxiang/training_runs/run_022b_v128_fullpika_root_retune_conservative_from_peak/probe_a_full_model/latest.pt"
V126_MICRO="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
V12_PEAK="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

run_one() {
  local label="$1"; local ckpt="$2"; local device="$3"
  local out="$OUT_BASE/$label"
  mkdir -p "$out"
  echo "[$label] start $(date +%H:%M:%S) on $device"
  "$PY" -u tools/external_arena.py \
    --checkpoint "$ckpt" \
    --device "$device" \
    --our-side black \
    --opening-suite-path "$OPENING_SUITE" \
    --max-openings 12 \
    --games-per-opening 2 \
    --games 999 \
    --parallel-games 2 \
    --cross-game-batch-cap 512 \
    --opp-engine pikafish \
    --opp-depth "$DEPTH" \
    --opp-threads 1 \
    --opp-hash-mb 64 \
    --seed "$SEED" \
    --our-sims 6400 \
    --our-c-puct 1.25 \
    --our-q-weight 1.0 \
    --our-q-clip 1.0 \
    --our-temperature-move 0.1 \
    --output-dir "$out" 2>&1 | tee "$out/run.log" \
    | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
  echo "[$label] done $(date +%H:%M:%S)"
}

queue_cuda0() {
  run_one "v128_retrain" "$V128_RETRAIN" cuda:0
  run_one "v126_micro"   "$V126_MICRO"   cuda:0
}
queue_cuda1() {
  run_one "probe_b" "$PROBE_B"  cuda:1
  run_one "v12_peak" "$V12_PEAK" cuda:1
}

mkdir -p "$OUT_BASE"
queue_cuda0 > "$OUT_BASE/queue_cuda0.log" 2>&1 &
P0=$!
queue_cuda1 > "$OUT_BASE/queue_cuda1.log" 2>&1 &
P1=$!
echo "launched: cuda0_pid=$P0 cuda1_pid=$P1"
wait $P0 $P1
echo "OPTION A d5 COMPARE ALL DONE $(date +%H:%M:%S)"
