#!/usr/bin/env bash
# SMOKE: redo Stage-2 with d20 teacher, from clean Stage-1 (181000). Tiny cycle to verify the loop works.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_051a_v2redo_d20_smoke"
SPROOT="/home/laure/alphaxiang/selfplay_runs_v051a_smoke"

mkdir -p "$RUNDIR" "$SPROOT"
# Seed from clean Stage-1 (181000) — does NOT touch the original
if [[ ! -f "$RUNDIR/latest.pt" ]]; then
  echo "seeding $RUNDIR/latest.pt from Stage-1 181000..."
  cp "$STAGE1" "$RUNDIR/latest.pt"
fi

"$PY" -u tools/stage1_driver.py \
  --venv-python "$PY" \
  --repo "$REPO" \
  --training-output-dir "$RUNDIR" \
  --selfplay-root "$SPROOT" \
  --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
  --cycles 1 \
  --samples-per-cycle 120 \
  --train-steps-per-cycle 20 \
  --reset-buffer-on-first-cycle \
  --train-lr-schedule-max-steps 240000 \
  --distill-depth 6 \
  --distill-workers 12 \
  --vspika-opp-depth 3 \
  --vspika-games-per-batch 4 \
  --vspika-our-sims 128 \
  --vspika-parallel-games 4 \
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
  2>&1 | tee "$RUNDIR/smoke.log"
echo "SMOKE EXIT=$? $(date +%H:%M:%S)"
