#!/usr/bin/env bash
# CORRECTED v2: train from Stage-1 (181000) on the pre-labeled d20 pool with BOOTSTRAP DISABLED,
# so the d20 selfplay pool dominates the batch (high oracle_cov) instead of human data flooding it.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

# make sure no stale trainer is alive
pkill -9 -f 'xiangqi_train.py' 2>/dev/null || true
sleep 2

POOL="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"
STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_053_v2redo_d20_pooltrain_nobootstrap"

mkdir -p "$RUNDIR"
if [[ ! -f "$RUNDIR/latest.pt" ]]; then
  echo "seeding $RUNDIR/latest.pt from clean Stage-1 181000..."
  cp "$STAGE1" "$RUNDIR/latest.pt"
fi

"$PY" -u xiangqi_train.py \
  --foreground \
  --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
  --selfplay-dirs "$POOL" \
  --output-dir "$RUNDIR" \
  --resume-path "$RUNDIR/latest.pt" \
  --device cuda:0 \
  --max-steps 184000 \
  --lr-schedule-max-steps 300000 \
  --learning-rate 2e-4 \
  --log-interval-steps 50 \
  --eval-interval-steps 250 \
  --save-interval-steps 500 \
  --snapshot-interval-steps 500 \
  --disable-selfplay-run-quality-gate \
  --reset-selfplay-ingest-state-on-resume \
  --disable-bootstrap-mode \
  --bootstrap-human-floor 0.10 \
  --wdl-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --value-target-scale 0.9 \
  --use-oracle-value \
  --policy-oracle-alpha 0.5 \
  2>&1 | tee "$RUNDIR/pooltrain2.log"
echo "POOLTRAIN2 EXIT=$? $(date +%H:%M:%S)"
