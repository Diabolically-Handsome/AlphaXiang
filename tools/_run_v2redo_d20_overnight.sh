#!/usr/bin/env bash
# Overnight d20 Stage-2 redo (cyclic driver) WITH the verified buffer fix.
# Key fix: --replay-buffer-size 14000 + --bootstrap-human-floor 0.05 so buffer_fill->~1.0 and the
# d20 selfplay pool dominates the batch (oracle_cov ~0.6-0.7, vs the broken 0.02 before).
# Pre-loads the existing 9663 d20-labeled shards (selfplay-root has cycles c001-c004).
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"
pkill -9 -f 'xiangqi_train.py' 2>/dev/null || true
sleep 2

STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_055_v2redo_d20_overnight"
SPROOT="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"   # already holds c001-c004 d20-labeled (9663)

mkdir -p "$RUNDIR"
[[ -f "$RUNDIR/latest.pt" ]] || { echo "seeding from Stage-1 181000"; cp "$STAGE1" "$RUNDIR/latest.pt"; }

"$PY" -u tools/stage1_driver.py \
  --venv-python "$PY" \
  --repo "$REPO" \
  --training-output-dir "$RUNDIR" \
  --selfplay-root "$SPROOT" \
  --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
  --cycles 0 \
  --samples-per-cycle 4000 \
  --train-steps-per-cycle 1500 \
  --reset-buffer-on-first-cycle \
  --replay-buffer-size 14000 \
  --train-bootstrap-human-floor 0.05 \
  --train-lr-schedule-max-steps 300000 \
  --train-learning-rate 2e-4 \
  --distill-depth 6 \
  --distill-workers 12 \
  --vspika-opp-depth 3 \
  --vspika-noise-ratio 0.15 \
  --vspika-our-sims 256 \
  --vspika-games-per-batch 40 \
  --vspika-parallel-games 8 \
  --device cuda:0 \
  --train-device cuda:0 \
  --selfplay-device cuda:0 \
  --oracle-label \
  --oracle-depth 20 \
  --oracle-workers 16 \
  --policy-oracle-label \
  --policy-oracle-depth 20 \
  --policy-oracle-multipv 6 \
  --policy-oracle-alpha 0.5 \
  --sanity-probe-every 2 \
  --sanity-probe-opp-depth 3 \
  --sanity-probe-our-sims 800 \
  2>&1 | tee -a "$RUNDIR/driver.log"
echo "OVERNIGHT DRIVER EXIT=$? $(date +%H:%M:%S)"
