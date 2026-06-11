#!/usr/bin/env bash
# v126_micro vs Pika d5 on cuda:1 (5080), identical protocol to the Option A compare.
# Separate output dir to avoid colliding with the cuda:0 queue's eventual v126_micro run.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"

OUT="/home/laure/alphaxiang/v128_optionA_d5_compare/v126_micro_5080"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
mkdir -p "$OUT"

echo "[v126_micro_5080] start $(date +%H:%M:%S)"
"$PY" -u tools/external_arena.py \
  --checkpoint "$CKPT" \
  --device cuda:1 \
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
echo "[v126_micro_5080] done $(date +%H:%M:%S)"
