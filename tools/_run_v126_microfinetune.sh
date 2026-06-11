#!/bin/bash
# v12.6-micro Path D: value-focused micro-finetune from v12 PEAK
# - Resumes from v12 latest.pt (step 286000)
# - Trains 10000 more steps to step 296000
# - Bumped value-loss-weight (2.0 vs default 0.5) to address middle-game value calibration
# - Includes failure slice as additional selfplay data
# - Uses oracle_value, oracle_policy, teacher_q
# Wall: ~24h (5000-10000 steps depending on lr_schedule)

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"

# Source: v12 PEAK at step 286000 (best snapshot, full training state)
# NOT v12 latest.pt at step 292500, which is past PEAK and possibly regressed.
V12_LATEST="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

# New training dir for v12.6-micro (does NOT overwrite v12)
NEW_TRAIN_DIR="/home/laure/alphaxiang/training_runs/run_017_v126_micro"

# Selfplay parent dirs (xiangqi_train accepts multiple parent roots,
# each containing one or more "run" subdirs with manifest.json + train/)
V12_SELFPLAY="/home/laure/alphaxiang/selfplay_runs_stage2_v12"
FAILSLICE_PARENT="/home/laure/alphaxiang/v126_microfinetune_data"

# Setup: create new training dir, seed it with v12 PEAK
mkdir -p "$NEW_TRAIN_DIR"
if [ ! -f "$NEW_TRAIN_DIR/latest.pt" ]; then
    echo "Seeding $NEW_TRAIN_DIR/latest.pt from v12 latest..."
    cp "$V12_LATEST" "$NEW_TRAIN_DIR/latest.pt"
fi

# Setup: create selfplay parent for failure slice
mkdir -p "$FAILSLICE_PARENT"
if [ ! -L "$FAILSLICE_PARENT/d4_slice" ]; then
    echo "Symlinking failure slice into selfplay parent..."
    ln -sf /home/laure/alphaxiang/v126_day3_d4_slice "$FAILSLICE_PARENT/d4_slice"
fi

cd "$REPO"

# Launch xiangqi_train with v12.6-micro config
"$PY" -u xiangqi_train.py \
    --foreground \
    --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
    --selfplay-dirs "$V12_SELFPLAY" "$FAILSLICE_PARENT" \
    --output-dir "$NEW_TRAIN_DIR" \
    --resume-path "$NEW_TRAIN_DIR/latest.pt" \
    --device cuda:0 \
    --max-steps 296000 \
    --lr-schedule-max-steps 300000 \
    --learning-rate 5e-5 \
    --log-interval-steps 100 \
    --eval-interval-steps 500 \
    --save-interval-steps 1000 \
    --snapshot-interval-steps 1000 \
    --disable-selfplay-run-quality-gate \
    --bootstrap-human-floor 0.05 \
    --wdl-loss-weight 1.0 \
    --value-loss-weight 2.0 \
    --value-target-scale 0.9 \
    --use-oracle-value \
    --policy-oracle-alpha 0.5 \
    --teacher-q-loss-weight 0.15 \
    --teacher-q-temperature-cp 80 \
    2>&1 | tee "$NEW_TRAIN_DIR/microfinetune.log"
echo "v12.6-micro training DONE"
