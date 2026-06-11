#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CHECKPOINT TAG [DEVICE=cuda:0]" >&2
  exit 2
fi

cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"

CKPT="$1"
TAG="$2"
DEVICE="${3:-cuda:0}"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
OUT_ROOT="/home/laure/alphaxiang/v14a_snapshot_smoke/${TAG}"

run_arena() {
  local name="$1"
  local depth="$2"
  local games="$3"
  local side="$4"
  local mode="$5"
  shift 5
  "$PY" -u tools/external_arena.py \
    --checkpoint "$CKPT" \
    --output-dir "${OUT_ROOT}/${name}_${mode}" \
    --games "$games" \
    --our-side "$side" \
    --parallel-games "$games" \
    --seed "$((202605110 + depth * 100 + games))" \
    --device "$DEVICE" \
    --opp-engine pikafish \
    --opp-depth "$depth" \
    --opp-threads 1 \
    --opp-hash-mb 256 \
    --our-sims 8000 \
    --our-q-weight 1.0 \
    --our-temperature-move 0.02 \
    --max-plies 180 \
    "$@"
}

# Model-only: no symbolic root guard or leaf mate extension.
for depth in 3 4 5; do
  run_arena "pika_d${depth}_g6" "$depth" 6 alternate model_only
done

# Ship config: current V13.3 inference safety stack.
for depth in 5 6; do
  run_arena "pika_d${depth}_g6" "$depth" 6 alternate ship \
    --our-root-mate1-blunder-guard \
    --our-tactical-mate1-extension \
    --our-tactical-mate2-extension
done
