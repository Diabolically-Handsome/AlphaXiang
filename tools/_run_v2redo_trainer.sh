#!/usr/bin/env bash
# Overnight TRAINER (GPU0): train from Stage-1 (181000) on a FROZEN copy of the d20 pool.
# Verified approach (pooltrain3): --replay-buffer-size = pool size + bootstrap floor 0.05 => oracle_cov ~0.95.
# Frozen copy so the parallel builder (which grows the LIVE pool) can't inject unlabeled shards here.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"
pkill -9 -f 'run_056_v2redo' 2>/dev/null || true

LIVE="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"
FROZEN="/home/laure/alphaxiang/selfplay_runs_v2redo_d20_frozen"
STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer"

# Freeze a copy of the currently-labeled pool (cycles c001-c004 = 9663 labeled)
if [[ ! -d "$FROZEN" ]]; then
  echo "freezing pool copy -> $FROZEN"
  mkdir -p "$FROZEN"
  for d in "$LIVE"/stage1_c00[1-4]_*; do cp -r "$d" "$FROZEN"/ 2>/dev/null || true; done
fi

mkdir -p "$RUNDIR"
[[ -f "$RUNDIR/latest.pt" ]] || { echo "seeding from Stage-1 181000"; cp "$STAGE1" "$RUNDIR/latest.pt"; }

"$PY" -u xiangqi_train.py \
  --foreground \
  --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
  --selfplay-dirs "$FROZEN" \
  --output-dir "$RUNDIR" \
  --resume-path "$RUNDIR/latest.pt" \
  --device cuda:0 \
  --max-steps 187000 \
  --lr-schedule-max-steps 300000 \
  --learning-rate 2e-4 \
  --replay-buffer-size 9663 \
  --log-interval-steps 50 \
  --eval-interval-steps 250 \
  --save-interval-steps 500 \
  --snapshot-interval-steps 500 \
  --disable-selfplay-run-quality-gate \
  --reset-selfplay-ingest-state-on-resume \
  --bootstrap-human-floor 0.05 \
  --wdl-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --value-target-scale 0.9 \
  --use-oracle-value \
  --policy-oracle-alpha 0.5 \
  2>&1 | tee -a "$RUNDIR/trainer.log"
echo "TRAINER EXIT=$? $(date +%H:%M:%S)"
