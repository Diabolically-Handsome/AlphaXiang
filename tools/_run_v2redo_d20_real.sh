#!/usr/bin/env bash
# REAL overnight run: redo Stage-2 with full-blood Pikafish d20 teacher, from clean Stage-1 (181000).
# Serial cuda:0 (the smoke-verified path). Runs forever (cycles=0) until killed / morning.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_051_v2redo_d20_from_stage1"
SPROOT="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"

mkdir -p "$RUNDIR" "$SPROOT"
if [[ ! -f "$RUNDIR/latest.pt" ]]; then
  echo "seeding $RUNDIR/latest.pt from clean Stage-1 181000..."
  cp "$STAGE1" "$RUNDIR/latest.pt"
fi

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
  --train-lr-schedule-max-steps 300000 \
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
  --sanity-probe-opp-depth 3 \
  --sanity-probe-our-sims 800 \
  2>&1 | tee "$RUNDIR/driver.log"
echo "DRIVER EXIT=$? $(date +%H:%M:%S)"
