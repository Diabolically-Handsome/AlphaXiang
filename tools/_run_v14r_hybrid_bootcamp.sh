#!/bin/bash
# V14R trunk-native CNN+Transformer hybrid bootcamp.
#
# This is a new mainline experiment, not a continuation of V14A-E.  The model is
# trained from scratch with a non-zero local CNN stem; the V13 checkpoint is used
# only as a short annealed teacher/anchor to get past the opening novice phase.

set -euo pipefail

REPO="${V14R_REPO:-/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer}"
PY="${V14R_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"

OUT_DIR="${V14R_OUT_DIR:-/home/laure/alphaxiang/training_runs/run_050a_v14r_200m_hybrid_bootcamp}"
TEACHER_CKPT="${V14R_TEACHER_CKPT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
RESUME_PATH="${V14R_RESUME_PATH:-}"
HUMAN_DATA_DIR="${V14R_HUMAN_DATA_DIR:-/home/laure/alphaxiang/human_bootstrap_data_elite_wdl}"
DEVICE="${V14R_DEVICE:-cuda:0}"
MAX_STEPS="${V14R_MAX_STEPS:-20000}"
LR="${V14R_LR:-1.5e-4}"
MICRO_BATCH="${V14R_MICRO_BATCH:-48}"
GRAD_ACCUM="${V14R_GRAD_ACCUM:-4}"
SAVE_INTERVAL="${V14R_SAVE_INTERVAL:-1000}"
SNAPSHOT_INTERVAL="${V14R_SNAPSHOT_INTERVAL:-1000}"
EVAL_INTERVAL="${V14R_EVAL_INTERVAL:-1000}"
TEACHER_ANNEAL_STEPS="${V14R_TEACHER_ANNEAL_STEPS:-5000}"
TEACHER_POLICY_KL="${V14R_TEACHER_POLICY_KL:-0.20}"
TEACHER_VALUE_MSE="${V14R_TEACHER_VALUE_MSE:-0.00}"
DIR_RATIOS="${V14R_DIR_RATIOS:-0.70 0.10 0.08 0.06 0.06}"
SEED="${V14R_SEED:-140501}"

if [ ! -f "$TEACHER_CKPT" ]; then
  echo "missing V14R teacher checkpoint: $TEACHER_CKPT" >&2
  exit 1
fi
if [ ! -d "$HUMAN_DATA_DIR" ]; then
  echo "missing human bootstrap data dir: $HUMAN_DATA_DIR" >&2
  exit 1
fi
if [ -n "$RESUME_PATH" ] && [ ! -f "$RESUME_PATH" ]; then
  echo "missing V14R resume checkpoint: $RESUME_PATH" >&2
  exit 1
fi

cd "$REPO"

RESUME_ARGS=()
if [ -n "$RESUME_PATH" ]; then
  RESUME_ARGS=(--resume-path "$RESUME_PATH")
fi

"$PY" xiangqi_train.py \
  "${RESUME_ARGS[@]}" \
  --model-preset v14r_200m_hybrid \
  --human-data-dir "$HUMAN_DATA_DIR" \
  --teacher-checkpoint "$TEACHER_CKPT" \
  --anchor-policy-kl-weight "$TEACHER_POLICY_KL" \
  --anchor-value-mse-weight "$TEACHER_VALUE_MSE" \
  --anchor-anneal-steps "$TEACHER_ANNEAL_STEPS" \
  --selfplay-dirs \
    /home/laure/alphaxiang/selfplay_runs_stage2_v12 \
    /home/laure/alphaxiang/v126_day3_d4_slice \
    /home/laure/alphaxiang/v133_p6_fullpika_round2_black_d5_verified_blunders_teacherq_d18 \
    /home/laure/alphaxiang/v133_p6_fullpika_round3_black_d6_verified_blunders_teacherq_d20 \
    /home/laure/alphaxiang/v133_p6_fullpika_round4_black_d6_verified_blunders_teacherq_d20 \
  --selfplay-dir-sampling-ratios $DIR_RATIOS \
  --output-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --foreground \
  --learning-rate "$LR" \
  --warmup-steps 1000 \
  --max-steps "$MAX_STEPS" \
  --lr-schedule-max-steps "$MAX_STEPS" \
  --save-interval-steps "$SAVE_INTERVAL" \
  --snapshot-interval-steps "$SNAPSHOT_INTERVAL" \
  --eval-interval-steps "$EVAL_INTERVAL" \
  --log-interval-steps 50 \
  --micro-batch-size "$MICRO_BATCH" \
  --grad-accum-steps "$GRAD_ACCUM" \
  --samples-per-unit 64 \
  --wdl-loss-weight 1.0 \
  --policy-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --wdl-value-consistency-weight 0.02 \
  --value-target-scale 0.9 \
  --policy-oracle-alpha 0.03 \
  --teacher-q-loss-weight 0.01 \
  --teacher-q-pairwise-loss-weight 0.02 \
  --teacher-q-temperature-cp 80.0 \
  --teacher-q-pairwise-min-gap-cp 80.0 \
  --seed "$SEED"
