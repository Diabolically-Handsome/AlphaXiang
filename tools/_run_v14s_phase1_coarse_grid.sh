#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"

PY="/home/laure/alphaxiang/venv_nospace/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt"
OUT="${V14S_OUT:-/home/laure/alphaxiang/v14s_search_tuning/phase1_coarse_d5_black}"
DEVICE="${V14S_DEVICE:-cuda:0}"
GAMES="${V14S_GAMES:-2}"

exec "$PY" -u tools/v14s_search_tuning_grid.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --python "$PY" \
  --device "$DEVICE" \
  --games "$GAMES" \
  --opening-suite-path "${V14S_OPENING_SUITE:-}" \
  --games-per-opening "${V14S_GAMES_PER_OPENING:-2}" \
  --max-openings "${V14S_MAX_OPENINGS:-0}" \
  --parallel-games 1 \
  --our-side black \
  --opp-depth 5 \
  --sims "${V14S_SIMS:-8000}" \
  --c-puct "${V14S_CPUCT:-1.0,1.25,1.5}" \
  --c-puct-base "${V14S_CPUCT_BASE:-1.0}" \
  --c-puct-factor "${V14S_CPUCT_FACTOR:-0.0}" \
  --q-weight "${V14S_Q_WEIGHT:-0.9,1.0,1.1}" \
  --q-clip "${V14S_Q_CLIP:-1.0}" \
  --fpu-reduction-root "${V14S_FPU_ROOT:--1.0}" \
  --fpu-reduction-tree "${V14S_FPU_TREE:--1.0}" \
  --temperature-move "${V14S_TEMP:-0.02}" \
  --enable-ship-safety
