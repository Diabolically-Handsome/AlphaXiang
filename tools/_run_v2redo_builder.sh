#!/usr/bin/env bash
# Overnight BUILDER (GPU1): grow the d20-labeled pool. Cyclic driver with negligible training
# (train-steps-per-cycle 1) used purely to generate+label d20 shards into the LIVE pool.
# Each cycle appends ~4000 d20-labeled samples to selfplay_runs_v2redo_d20.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_057_v2redo_builder"
SPROOT="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"   # LIVE pool (grows here)

mkdir -p "$RUNDIR"
[[ -f "$RUNDIR/latest.pt" ]] || cp "$STAGE1" "$RUNDIR/latest.pt"

"$PY" -u tools/stage1_driver.py \
  --venv-python "$PY" \
  --repo "$REPO" \
  --training-output-dir "$RUNDIR" \
  --selfplay-root "$SPROOT" \
  --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
  --cycles 0 \
  --samples-per-cycle 4000 \
  --train-steps-per-cycle 1 \
  --reset-buffer-on-first-cycle \
  --train-replay-buffer-size 8000 \
  --train-bootstrap-human-floor 0.05 \
  --distill-depth 6 \
  --distill-workers 10 \
  --vspika-opp-depth 3 \
  --vspika-noise-ratio 0.15 \
  --vspika-our-sims 256 \
  --vspika-games-per-batch 40 \
  --vspika-parallel-games 6 \
  --device cuda:1 \
  --train-device cuda:1 \
  --selfplay-device cuda:1 \
  --oracle-label \
  --oracle-depth 20 \
  --oracle-workers 14 \
  --policy-oracle-label \
  --policy-oracle-depth 20 \
  --policy-oracle-multipv 6 \
  --policy-oracle-alpha 0.5 \
  --sanity-probe-every 0 \
  2>&1 | tee -a "$RUNDIR/builder.log"
echo "BUILDER EXIT=$? $(date +%H:%M:%S)"
