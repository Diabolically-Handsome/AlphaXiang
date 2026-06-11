#!/usr/bin/env bash
# RESUME (no reset, no reseed): used by the watchdog to restart the d20 Stage-2 redo
# after a crash. latest.pt already holds progress + valid buffer refs to this run's own
# d20 shards, so we resume normally and KEEP the accumulated buffer (no --reset-buffer-on-first-cycle).
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

RUNDIR="/home/laure/alphaxiang/training_runs/run_051_v2redo_d20_from_stage1"
SPROOT="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"

if [[ ! -f "$RUNDIR/latest.pt" ]]; then
  echo "[resume] ERROR: $RUNDIR/latest.pt missing; refusing to resume" >&2
  exit 3
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
  2>&1 | tee -a "$RUNDIR/driver.log"
echo "[resume] DRIVER EXIT=$? $(date +%H:%M:%S)"
