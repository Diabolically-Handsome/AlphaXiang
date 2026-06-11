#!/usr/bin/env bash
# CORRECTED approach: train from clean Stage-1 (181000) directly on the PRE-LABELED d20 pool.
# Pre-labeled shards => ingestion loads d20 oracle_value/oracle_policy => high oracle_cov (fixes the staleness bug).
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

POOL="/home/laure/alphaxiang/selfplay_runs_v2redo_d20"
PENDING="/home/laure/alphaxiang/selfplay_runs_v2redo_d20_pending"
STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_052_v2redo_d20_pooltrain"

# Move the UNLABELED cycle-5 dirs out of the pool so training sees only fully-d20-labeled shards
mkdir -p "$PENDING"
for d in "$POOL"/stage1_c005_*; do
  [[ -e "$d" ]] && mv "$d" "$PENDING"/ 2>/dev/null || true
done

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
  --max-steps 183000 \
  --lr-schedule-max-steps 300000 \
  --learning-rate 2e-4 \
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
  2>&1 | tee "$RUNDIR/pooltrain.log"
echo "POOLTRAIN EXIT=$? $(date +%H:%M:%S)"
