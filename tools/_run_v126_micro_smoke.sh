#!/bin/bash
# Smoke test for v12.6-micro training: 50 steps from v12 PEAK
# Throwaway dir; will be deleted after.
set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
V12_LATEST="/home/laure/alphaxiang/training_runs/run_016_stage2_v12/latest.pt"
SMOKE_DIR="/home/laure/alphaxiang/training_runs/run_018_v126_smoke"
V12_SELFPLAY="/home/laure/alphaxiang/selfplay_runs_stage2_v12"
FAILSLICE_PARENT="/home/laure/alphaxiang/v126_microfinetune_data"

mkdir -p "$SMOKE_DIR" "$FAILSLICE_PARENT"
[ ! -L "$FAILSLICE_PARENT/d4_slice" ] && ln -sf /home/laure/alphaxiang/v126_day3_d4_slice "$FAILSLICE_PARENT/d4_slice"
[ ! -f "$SMOKE_DIR/latest.pt" ] && cp "$V12_LATEST" "$SMOKE_DIR/latest.pt"

cd "$REPO"
"$PY" -u xiangqi_train.py \
    --foreground \
    --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
    --selfplay-dirs "$V12_SELFPLAY" "$FAILSLICE_PARENT" \
    --output-dir "$SMOKE_DIR" \
    --resume-path "$SMOKE_DIR/latest.pt" \
    --device cuda:0 \
    --max-steps 292600 \
    --lr-schedule-max-steps 300000 \
    --learning-rate 5e-5 \
    --log-interval-steps 10 \
    --eval-interval-steps 50 \
    --save-interval-steps 50 \
    --snapshot-interval-steps 50 \
    --disable-selfplay-run-quality-gate \
    --bootstrap-human-floor 0.05 \
    --wdl-loss-weight 1.0 \
    --value-loss-weight 2.0 \
    --value-target-scale 0.9 \
    --use-oracle-value \
    --policy-oracle-alpha 0.5 \
    --teacher-q-loss-weight 0.15 \
    --teacher-q-temperature-cp 80 \
    2>&1 | tee "$SMOKE_DIR/smoke.log"
echo "smoke DONE"
