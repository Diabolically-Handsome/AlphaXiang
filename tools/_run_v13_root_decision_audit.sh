#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${V13_ROOT_AUDIT_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

CHECKPOINT="${CHECKPOINT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
OUT_DIR="${OUT_DIR:-/home/laure/alphaxiang/v13_root_decision_audit}"
ARENA_JSON="${ARENA_JSON:-/home/laure/alphaxiang/v13_ab_search_diagnostic/micro_20260519_081251/micro_d5_mcts_baseline/external_arena_20260519_084340.json}"
OPENING_SUITE="${OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"

MAX_POSITIONS="${MAX_POSITIONS:-32}"
MAX_POSITIONS_PER_FILE="${MAX_POSITIONS_PER_FILE:-0}"
PLY_STRIDE="${PLY_STRIDE:-4}"
MCTS_SIMS="${MCTS_SIMS:-8000}"
PIKA_ROOT_DEPTH="${PIKA_ROOT_DEPTH:-12}"
PIKA_CHILD_DEPTH="${PIKA_CHILD_DEPTH:-14}"
PIKA_WORKERS="${PIKA_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
TAG="${TAG:-$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"

"$PY" tools/v13_root_decision_audit.py "$ARENA_JSON" \
  --checkpoint "$CHECKPOINT" \
  --opening-suite-path "$OPENING_SUITE" \
  --out-json "$OUT_DIR/root_decision_audit_${TAG}.json" \
  --out-md "$OUT_DIR/root_decision_audit_${TAG}.md" \
  --results opp_win \
  --only-side black \
  --max-positions "$MAX_POSITIONS" \
  --max-positions-per-file "$MAX_POSITIONS_PER_FILE" \
  --ply-stride "$PLY_STRIDE" \
  --raw-policy-top-k 16 \
  --mcts-top-k 16 \
  --mcts-sims "$MCTS_SIMS" \
  --mcts-c-puct 1.45 \
  --mcts-q-weight 1.0 \
  --mcts-temperature-move 0.02 \
  --pika-root-depth "$PIKA_ROOT_DEPTH" \
  --pika-child-depth "$PIKA_CHILD_DEPTH" \
  --pika-root-multipv 8 \
  --pika-workers "$PIKA_WORKERS" \
  --pika-hash-mb 128 \
  --device "$DEVICE"
