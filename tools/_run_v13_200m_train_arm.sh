#!/bin/bash
# v13 200M train entrypoint.  Use V13_ARM=dense or strategy.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"

ARM="${V13_ARM:-dense}"
case "$ARM" in
  dense)
    PRESET="v13_200m_dense"
    OUT_DIR="${V13_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_020a_v13_200m_dense_baseline}"
    SEED="${V13_SEED:-132001}"
    ;;
  strategy)
    PRESET="v13_200m_strategy"
    OUT_DIR="${V13_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_020b_v13_200m_strategy_tokens}"
    SEED="${V13_SEED:-132002}"
    ;;
  *)
    echo "V13_ARM must be dense or strategy, got: $ARM" >&2
    exit 2
    ;;
esac

TEACHER_CKPT="${V13_TEACHER_CKPT:-/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt}"
HUMAN_DATA_DIR="${V13_HUMAN_DATA_DIR:-/home/laure/alphaxiang/human_bootstrap_data_elite_wdl}"
DEVICE="${V13_DEVICE:-cuda:0}"
MAX_STEPS="${V13_MAX_STEPS:-300000}"
LR="${V13_LR:-2e-4}"
MICRO_BATCH="${V13_MICRO_BATCH:-64}"
GRAD_ACCUM="${V13_GRAD_ACCUM:-4}"
SAVE_INTERVAL="${V13_SAVE_INTERVAL:-5000}"
SNAPSHOT_INTERVAL="${V13_SNAPSHOT_INTERVAL:-5000}"
EVAL_INTERVAL="${V13_EVAL_INTERVAL:-5000}"
TEACHER_ANNEAL_STEPS="${V13_TEACHER_ANNEAL_STEPS:-50000}"
TEACHER_POLICY_KL="${V13_TEACHER_POLICY_KL:-0.35}"
TEACHER_VALUE_MSE="${V13_TEACHER_VALUE_MSE:-0.03}"
DIR_RATIOS="${V13_DIR_RATIOS:-0.82 0.12 0.06}"

if [ ! -f "$TEACHER_CKPT" ]; then
  echo "missing v13 bootstrap teacher checkpoint: $TEACHER_CKPT" >&2
  exit 1
fi
if [ ! -d "$HUMAN_DATA_DIR" ]; then
  echo "missing human bootstrap data dir: $HUMAN_DATA_DIR" >&2
  exit 1
fi

cd "$REPO"

"$PY" xiangqi_train.py \
  --model-preset "$PRESET" \
  --human-data-dir "$HUMAN_DATA_DIR" \
  --teacher-checkpoint "$TEACHER_CKPT" \
  --anchor-policy-kl-weight "$TEACHER_POLICY_KL" \
  --anchor-value-mse-weight "$TEACHER_VALUE_MSE" \
  --anchor-anneal-steps "$TEACHER_ANNEAL_STEPS" \
  --selfplay-dirs \
    /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
    /home/laure/alphaxiang/v126_day3_d4_slice \
    /home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses \
  --selfplay-dir-sampling-ratios $DIR_RATIOS \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --foreground \
  --learning-rate "$LR" \
  --warmup-steps 2000 \
  --max-steps "$MAX_STEPS" \
  --lr-schedule-max-steps "$MAX_STEPS" \
  --save-interval-steps "$SAVE_INTERVAL" \
  --snapshot-interval-steps "$SNAPSHOT_INTERVAL" \
  --eval-interval-steps "$EVAL_INTERVAL" \
  --log-interval-steps 100 \
  --micro-batch-size "$MICRO_BATCH" \
  --grad-accum-steps "$GRAD_ACCUM" \
  --samples-per-unit 64 \
  --wdl-loss-weight 1.0 \
  --policy-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --wdl-value-consistency-weight 0.02 \
  --value-target-scale 0.9 \
  --policy-oracle-alpha 0.03 \
  --teacher-q-loss-weight 0.03 \
  --teacher-q-temperature-cp 80.0 \
  --seed "$SEED"
