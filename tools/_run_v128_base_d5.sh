#!/usr/bin/env bash
# Real V12.8 candidate: base "more_conservative" (79/21 rehearsal mix) vs Pika d5 on cuda:0.
# Identical protocol/seed to the Option A compare so it's directly comparable to v126_micro.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"

OUT="/home/laure/alphaxiang/v128_optionA_d5_compare/v128_base_more_conservative"
CKPT="/home/laure/alphaxiang/training_runs/run_025_v128_d20_stage2_20k_smoke/full_model_200step_more_conservative/latest.pt"
mkdir -p "$OUT"

echo "[v128_base] start $(date +%H:%M:%S)"
"$PY" -u tools/external_arena.py \
  --checkpoint "$CKPT" \
  --device cuda:0 \
  --our-side black \
  --opening-suite-path "arena_openings/human_val_opening_suite_v1.json" \
  --max-openings 12 \
  --games-per-opening 2 \
  --games 999 \
  --parallel-games 2 \
  --cross-game-batch-cap 512 \
  --opp-engine pikafish \
  --opp-depth 5 \
  --opp-threads 1 \
  --opp-hash-mb 64 \
  --seed 2026060101 \
  --our-sims 6400 \
  --our-c-puct 1.25 \
  --our-q-weight 1.0 \
  --our-q-clip 1.0 \
  --our-temperature-move 0.1 \
  --output-dir "$OUT" 2>&1 | tee "$OUT/run.log" \
  | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|loaded our model' || true
echo "[v128_base] done $(date +%H:%M:%S)"
