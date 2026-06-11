#!/bin/bash
# Self-play RL closed loop, seeded from geo. Pure-ish AlphaZero:
#   self-play @3200 sims (value target = game outcome z), train, arena-gate every
#   7 cycles @12800 vs previous best (promote at >=55%). human mix 0; light human
#   floor as a stabilizer. Env-overridable for smoke (CYCLES/SP_SAMPLES/TRAIN_STEPS/EXTRA).
set -euo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
GEO="/home/laure/alphaxiang/training_runs/run_063_attn_geo/latest.pt"

OUT_DIR="${OUT_DIR:-/home/laure/alphaxiang/training_runs/run_070_selfplay_rl}"
SP_ROOT="${SP_ROOT:-/home/laure/alphaxiang/selfplay_rl_run070}"
ARENA_ROOT="${ARENA_ROOT:-/home/laure/alphaxiang/arena_runs/rl_run070}"
DEVICE="${DEVICE:-cuda:0}"
CYCLES="${CYCLES:-0}"
SP_SIMS="${SP_SIMS:-3200}"
SP_SAMPLES="${SP_SAMPLES:-10000}"
SP_WORKERS="${SP_WORKERS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-1500}"
MICRO_BATCH="${MICRO_BATCH:-256}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
HUMAN_FLOOR="${HUMAN_FLOOR:-0.05}"
LR_SCHED="${LR_SCHED:-600000}"
ARENA_SIMS="${ARENA_SIMS:-12800}"
ARENA_GAMES="${ARENA_GAMES:-24}"
ARENA_EVERY="${ARENA_EVERY:-7}"
ARENA_THRESH="${ARENA_THRESH:-0.55}"
KEEP_BEST="${KEEP_BEST:-3}"
KEEP_LATEST="${KEEP_LATEST:-3}"
SEED="${SEED:-70007}"
# B-experiment: policy anchor to geo (KL). ANCHOR_KL=0 => off (default). >0 => anchor on.
ANCHOR_CKPT="${ANCHOR_CKPT:-$GEO}"
ANCHOR_KL="${ANCHOR_KL:-0}"
ANCHOR_ANNEAL="${ANCHOR_ANNEAL:-0}"
# v16 frozen-evaluator mode: set FROZEN_VALUE=$GEO (or any ckpt) to run self-play
# and gates as policy/value chimeras + policy-only training + optimizer reset.
FROZEN_VALUE="${FROZEN_VALUE:-}"
EXTRA="${EXTRA:-}"

if [ ! -f "$GEO" ]; then echo "missing geo seed: $GEO" >&2; exit 1; fi
mkdir -p "$OUT_DIR"
if [ ! -f "$OUT_DIR/latest.pt" ]; then
  echo "seeding geo -> $OUT_DIR/{latest,best}.pt"
  cp "$GEO" "$OUT_DIR/latest.pt"
  cp "$GEO" "$OUT_DIR/best.pt"
fi

cd "$REPO"
"$PY" xiangqi_closed_loop.py \
  --training-output-dir "$OUT_DIR" \
  --selfplay-output-root "$SP_ROOT" \
  --arena-output-root "$ARENA_ROOT" \
  --device "$DEVICE" \
  --pause-at-local-time "" \
  --cycles "$CYCLES" \
  --selfplay-num-simulations "$SP_SIMS" \
  --selfplay-target-samples-per-cycle "$SP_SAMPLES" \
  --selfplay-num-workers "$SP_WORKERS" \
  --selfplay-human-position-mix-ratio 0 \
  --train-steps-per-cycle "$TRAIN_STEPS" \
  --train-micro-batch-size "$MICRO_BATCH" \
  --train-grad-accum-steps "$GRAD_ACCUM" \
  --train-bootstrap-human-floor "$HUMAN_FLOOR" \
  --train-lr-schedule-max-steps "$LR_SCHED" \
  --arena-sims "$ARENA_SIMS" \
  --arena-games "$ARENA_GAMES" \
  --arena-every-n-cycles "$ARENA_EVERY" \
  --arena-accept-threshold "$ARENA_THRESH" \
  --snapshot-keep-best-count "$KEEP_BEST" \
  --snapshot-keep-latest-count "$KEEP_LATEST" \
  --seed "$SEED" \
  --rl-anchor-checkpoint "$ANCHOR_CKPT" \
  --rl-anchor-policy-kl-weight "$ANCHOR_KL" \
  --rl-anchor-anneal-steps "$ANCHOR_ANNEAL" \
  ${FROZEN_VALUE:+--rl-frozen-value-checkpoint "$FROZEN_VALUE"} \
  $EXTRA 2>&1
