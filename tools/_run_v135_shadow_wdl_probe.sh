#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V135_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CHECKPOINT="${V135_CHECKPOINT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
OPENINGS="${V135_OPENINGS:-arena_openings/human_val_opening_suite_v1.json}"
OUT_ROOT="${V135_OUT_ROOT:-/home/laure/alphaxiang/v13_shadow_wdl_probe}"
DEVICE="${V135_DEVICE:-cuda:0}"
SEED="${V135_SEED:-20260418}"
DEPTH="${V135_PIKA_DEPTH:-5}"
SHADOW_SIMS="${V135_SHADOW_SIMS:-8000}"
OUR_SIMS="${V135_OUR_SIMS:-8000}"
SHADOW_VERIFIER_TOP_K="${V135_SHADOW_VERIFIER_TOP_K:-6}"
SHADOW_VERIFIER_MARGIN_CP="${V135_SHADOW_VERIFIER_MARGIN_CP:-100}"
SHADOW_VERIFIER_MATE_RISK_MARGIN_CP="${V135_SHADOW_VERIFIER_MATE_RISK_MARGIN_CP:--1}"
SHADOW_VERIFIER_MATE_RISK_CP="${V135_SHADOW_VERIFIER_MATE_RISK_CP:-19000}"
SHADOW_VERIFIER_ESCAPE_MARGIN_CP="${V135_SHADOW_VERIFIER_ESCAPE_MARGIN_CP:--1}"
SHADOW_VERIFIER_ESCAPE_RISK_CP="${V135_SHADOW_VERIFIER_ESCAPE_RISK_CP:-500}"
SHADOW_VERIFIER_ESCAPE_SAFE_CP="${V135_SHADOW_VERIFIER_ESCAPE_SAFE_CP:-100}"
VERIFIER_DEPTH="${V135_VERIFIER_DEPTH:-10}"
MAX_OPENINGS="${V135_MAX_OPENINGS:-12}"
GAMES_PER_OPENING="${V135_GAMES_PER_OPENING:-1}"
PARALLEL_GAMES="${V135_PARALLEL_GAMES:-1}"
CROSS_GAME_BATCHING="${V135_CROSS_GAME_BATCHING:-0}"
OPP_THREADS="${V135_OPP_THREADS:-1}"
OPP_HASH_MB="${V135_OPP_HASH_MB:-128}"
VERIFIER_THREADS="${V135_VERIFIER_THREADS:-1}"
VERIFIER_HASH_MB="${V135_VERIFIER_HASH_MB:-64}"
PHASE="${1:-${V135_PHASE:-d5}}"
SMOKE_OUT_ROOT="${V135_SMOKE_OUT_ROOT:-tmp_v135_shadow_wdl_runner_smoke}"

run_audit() {
  local input_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  "$PY" tools/v13_shadow_value_audit.py "$input_dir" \
    --out-json "$out_dir/shadow_value_audit.json" \
    --out-md "$out_dir/shadow_value_audit.md" \
    --early-ply-threshold "${V135_EARLY_PLY_THRESHOLD:-80}"
}

run_compare_and_decide() {
  local label="$1"
  local baseline_dir="$2"
  local gated_dir="$3"
  local min_games="$4"
  "$PY" tools/v13_shadow_gate_compare.py \
    --baseline "$baseline_dir" \
    --gated "$gated_dir" \
    --out-json "$OUT_ROOT/${label}_shadow_gate_compare.json" \
    --out-md "$OUT_ROOT/${label}_shadow_gate_compare.md"
  "$PY" tools/v13_shadow_gate_decision.py "$OUT_ROOT/${label}_shadow_gate_compare.json" \
    --out-json "$OUT_ROOT/${label}_shadow_gate_decision.json" \
    --out-md "$OUT_ROOT/${label}_shadow_gate_decision.md" \
    --min-paired-games "$min_games" \
    --min-score-delta "${V135_MIN_SCORE_DELTA:-0.0}" \
    --max-mate-loss-increase "${V135_MAX_MATE_LOSS_INCREASE:-0}" \
    --max-longcheck-loss-increase "${V135_MAX_LONGCHECK_LOSS_INCREASE:-0}" \
    --min-gate-events "${V135_MIN_GATE_EVENTS:-1}" \
    --max-worsened-games "${V135_MAX_WORSENED_GAMES:-0}"
}

common_args=(
  --checkpoint "$CHECKPOINT"
  --opening-suite-path "$OPENINGS"
  --max-openings "$MAX_OPENINGS"
  --games-per-opening "$GAMES_PER_OPENING"
  --our-side black
  --seed "$SEED"
  --parallel-games "$PARALLEL_GAMES"
  --our-sims "$OUR_SIMS"
  --our-c-puct 1.45
  --our-q-weight 1.0
  --our-temperature-move 0.02
  --our-value-source scalar
  --our-shadow-value-source wdl
  --our-shadow-sims "$SHADOW_SIMS"
  --our-shadow-top-k 8
  --our-shadow-side black
  --our-root-mate1-blunder-guard
  --our-tactical-mate1-extension
  --our-tactical-mate2-extension
  --opp-depth "$DEPTH"
  --opp-threads "$OPP_THREADS"
  --opp-hash-mb "$OPP_HASH_MB"
  --our-verifier-threads "$VERIFIER_THREADS"
  --our-verifier-hash-mb "$VERIFIER_HASH_MB"
  --device "$DEVICE"
)
if [[ "$CROSS_GAME_BATCHING" != "1" ]]; then
  common_args+=(--no-cross-game-batching)
fi

case "$PHASE" in
  help)
    cat <<EOF
V13.5 shadow WDL probe runner

Phases:
  static      Compile/check scripts only.
  smoke       CPU random-opponent end-to-end smoke: baseline, gated, audit, compare, decision.
  d5          Baseline scalar search with WDL shadow probe vs Pika d5.
  d5-gated    Same as d5, but scalar/WDL disagreement triggers Pikafish verifier.
  compare-d5  Compare d5 vs d5-gated and run decision gate.
  d5-full     Run static, d5, d5-gated, compare-d5.
  d6          Baseline scalar search with WDL shadow probe vs Pika d6.
  d6-gated    Same as d6, but scalar/WDL disagreement triggers Pikafish verifier.
  compare-d6  Compare d6 vs d6-gated and run decision gate.
  d6-full     Run static, d6, d6-gated, compare-d6.

Key env knobs:
  V135_OUT_ROOT=$OUT_ROOT
  V135_SEED=$SEED
  V135_OUR_SIMS=$OUR_SIMS
  V135_SHADOW_SIMS=$SHADOW_SIMS
  V135_SHADOW_VERIFIER_TOP_K=$SHADOW_VERIFIER_TOP_K
  V135_SHADOW_VERIFIER_MARGIN_CP=$SHADOW_VERIFIER_MARGIN_CP
  V135_SHADOW_VERIFIER_MATE_RISK_MARGIN_CP=$SHADOW_VERIFIER_MATE_RISK_MARGIN_CP
  V135_SHADOW_VERIFIER_MATE_RISK_CP=$SHADOW_VERIFIER_MATE_RISK_CP
  V135_SHADOW_VERIFIER_ESCAPE_MARGIN_CP=$SHADOW_VERIFIER_ESCAPE_MARGIN_CP
  V135_SHADOW_VERIFIER_ESCAPE_RISK_CP=$SHADOW_VERIFIER_ESCAPE_RISK_CP
  V135_SHADOW_VERIFIER_ESCAPE_SAFE_CP=$SHADOW_VERIFIER_ESCAPE_SAFE_CP
  V135_VERIFIER_DEPTH=$VERIFIER_DEPTH
  V135_PARALLEL_GAMES=$PARALLEL_GAMES
  V135_CROSS_GAME_BATCHING=$CROSS_GAME_BATCHING
  V135_OPP_THREADS=$OPP_THREADS
  V135_OPP_HASH_MB=$OPP_HASH_MB
  V135_VERIFIER_THREADS=$VERIFIER_THREADS
  V135_VERIFIER_HASH_MB=$VERIFIER_HASH_MB
  V135_MIN_SCORE_DELTA=${V135_MIN_SCORE_DELTA:-0.0}
  V135_MAX_MATE_LOSS_INCREASE=${V135_MAX_MATE_LOSS_INCREASE:-0}
  V135_MAX_LONGCHECK_LOSS_INCREASE=${V135_MAX_LONGCHECK_LOSS_INCREASE:-0}
  V135_MIN_GATE_EVENTS=${V135_MIN_GATE_EVENTS:-1}
  V135_MAX_WORSENED_GAMES=${V135_MAX_WORSENED_GAMES:-0}

Recommended CUDA run:
  bash tools/_run_v135_shadow_wdl_probe.sh d5-full
  # only if d5-full passes:
  bash tools/_run_v135_shadow_wdl_probe.sh d6-full
EOF
    ;;
  static)
    "$PY" -m py_compile \
      tools/external_arena.py \
      tools/v13_shadow_value_audit.py \
      tools/v13_shadow_gate_compare.py \
      tools/v13_shadow_gate_decision.py
    bash -n tools/_run_v135_shadow_wdl_probe.sh
    ;;
  smoke)
    "$PY" -m py_compile \
      tools/external_arena.py \
      tools/v13_shadow_value_audit.py \
      tools/v13_shadow_gate_compare.py \
      tools/v13_shadow_gate_decision.py
    "$PY" tools/external_arena.py \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$SMOKE_OUT_ROOT/baseline" \
      --games 1 \
      --parallel-games 1 \
      --no-cross-game-batching \
      --opp-random \
      --our-sims 2 \
      --our-c-puct 1.45 \
      --our-q-weight 1.0 \
      --our-temperature-move 0.02 \
      --our-value-source scalar \
      --our-shadow-value-source wdl \
      --our-shadow-sims 1 \
      --our-shadow-top-k 4 \
      --max-plies 2 \
      --device cpu \
      --disable-bf16
    "$PY" tools/external_arena.py \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$SMOKE_OUT_ROOT/gated" \
      --games 1 \
      --parallel-games 1 \
      --no-cross-game-batching \
      --opp-random \
      --our-sims 2 \
      --our-c-puct 1.45 \
      --our-q-weight 1.0 \
      --our-temperature-move 0.02 \
      --our-value-source scalar \
      --our-shadow-value-source wdl \
      --our-shadow-sims 1 \
      --our-shadow-top-k 4 \
      --our-shadow-disagreement-verifier \
      --our-shadow-verifier-top-k 4 \
      --our-shadow-verifier-margin-cp 1 \
      --our-verifier-depth 1 \
      --our-root-mate1-blunder-guard \
      --our-root-mate2-blunder-guard \
      --our-root-forcing-check-guard-plies 3 \
      --max-plies 2 \
      --device cpu \
      --disable-bf16
    "$PY" tools/v13_shadow_value_audit.py "$SMOKE_OUT_ROOT/baseline" \
      --out-json "$SMOKE_OUT_ROOT/baseline_shadow_audit.json" \
      --out-md "$SMOKE_OUT_ROOT/baseline_shadow_audit.md" \
      --early-ply-threshold 10
    "$PY" tools/v13_shadow_value_audit.py "$SMOKE_OUT_ROOT/gated" \
      --out-json "$SMOKE_OUT_ROOT/gated_shadow_audit.json" \
      --out-md "$SMOKE_OUT_ROOT/gated_shadow_audit.md" \
      --early-ply-threshold 10
    "$PY" tools/v13_shadow_gate_compare.py \
      --baseline "$SMOKE_OUT_ROOT/baseline" \
      --gated "$SMOKE_OUT_ROOT/gated" \
      --out-json "$SMOKE_OUT_ROOT/shadow_gate_compare.json" \
      --out-md "$SMOKE_OUT_ROOT/shadow_gate_compare.md"
    "$PY" tools/v13_shadow_gate_decision.py "$SMOKE_OUT_ROOT/shadow_gate_compare.json" \
      --out-json "$SMOKE_OUT_ROOT/shadow_gate_decision.json" \
      --out-md "$SMOKE_OUT_ROOT/shadow_gate_decision.md" \
      --min-paired-games 1 \
      --min-gate-events 0 \
      --max-worsened-games 0
    ;;
  d5)
    "$PY" tools/external_arena.py "${common_args[@]}" \
      --output-dir "$OUT_ROOT/d5_shadow"
    run_audit "$OUT_ROOT/d5_shadow" "$OUT_ROOT/d5_shadow_audit"
    ;;
  d6)
    "$PY" tools/external_arena.py "${common_args[@]}" \
      --opp-depth 6 \
      --output-dir "$OUT_ROOT/d6_shadow"
    run_audit "$OUT_ROOT/d6_shadow" "$OUT_ROOT/d6_shadow_audit"
    ;;
  d5-gated)
    "$PY" tools/external_arena.py "${common_args[@]}" \
      --our-shadow-disagreement-verifier \
      --our-shadow-verifier-top-k "$SHADOW_VERIFIER_TOP_K" \
      --our-shadow-verifier-margin-cp "$SHADOW_VERIFIER_MARGIN_CP" \
      --our-shadow-verifier-mate-risk-margin-cp "$SHADOW_VERIFIER_MATE_RISK_MARGIN_CP" \
      --our-shadow-verifier-mate-risk-cp "$SHADOW_VERIFIER_MATE_RISK_CP" \
      --our-shadow-verifier-escape-margin-cp "$SHADOW_VERIFIER_ESCAPE_MARGIN_CP" \
      --our-shadow-verifier-escape-risk-cp "$SHADOW_VERIFIER_ESCAPE_RISK_CP" \
      --our-shadow-verifier-escape-safe-cp "$SHADOW_VERIFIER_ESCAPE_SAFE_CP" \
      --our-verifier-depth "$VERIFIER_DEPTH" \
      --our-root-mate2-blunder-guard \
      --our-root-forcing-check-guard-plies "${V135_FORCING_CHECK_PLIES:-5}" \
      --output-dir "$OUT_ROOT/d5_shadow_gated"
    run_audit "$OUT_ROOT/d5_shadow_gated" "$OUT_ROOT/d5_shadow_gated_audit"
    ;;
  d6-gated)
    "$PY" tools/external_arena.py "${common_args[@]}" \
      --opp-depth 6 \
      --our-shadow-disagreement-verifier \
      --our-shadow-verifier-top-k "$SHADOW_VERIFIER_TOP_K" \
      --our-shadow-verifier-margin-cp "$SHADOW_VERIFIER_MARGIN_CP" \
      --our-shadow-verifier-mate-risk-margin-cp "$SHADOW_VERIFIER_MATE_RISK_MARGIN_CP" \
      --our-shadow-verifier-mate-risk-cp "$SHADOW_VERIFIER_MATE_RISK_CP" \
      --our-shadow-verifier-escape-margin-cp "$SHADOW_VERIFIER_ESCAPE_MARGIN_CP" \
      --our-shadow-verifier-escape-risk-cp "$SHADOW_VERIFIER_ESCAPE_RISK_CP" \
      --our-shadow-verifier-escape-safe-cp "$SHADOW_VERIFIER_ESCAPE_SAFE_CP" \
      --our-verifier-depth "$VERIFIER_DEPTH" \
      --our-root-mate2-blunder-guard \
      --our-root-forcing-check-guard-plies "${V135_FORCING_CHECK_PLIES:-5}" \
      --output-dir "$OUT_ROOT/d6_shadow_gated"
    run_audit "$OUT_ROOT/d6_shadow_gated" "$OUT_ROOT/d6_shadow_gated_audit"
    ;;
  audit-d5)
    run_audit "$OUT_ROOT/d5_shadow" "$OUT_ROOT/d5_shadow_audit"
    ;;
  audit-d6)
    run_audit "$OUT_ROOT/d6_shadow" "$OUT_ROOT/d6_shadow_audit"
    ;;
  compare-d5)
    run_compare_and_decide "d5" "$OUT_ROOT/d5_shadow" "$OUT_ROOT/d5_shadow_gated" "$(( MAX_OPENINGS * GAMES_PER_OPENING ))"
    ;;
  compare-d6)
    run_compare_and_decide "d6" "$OUT_ROOT/d6_shadow" "$OUT_ROOT/d6_shadow_gated" "$(( MAX_OPENINGS * GAMES_PER_OPENING ))"
    ;;
  d5-full)
    "$0" static
    "$0" d5
    "$0" d5-gated
    "$0" compare-d5
    ;;
  d6-full)
    "$0" static
    "$0" d6
    "$0" d6-gated
    "$0" compare-d6
    ;;
  *)
    printf 'unknown V135_PHASE: %s\n' "$PHASE" >&2
    printf 'valid phases: help, static, smoke, d5, d6, d5-gated, d6-gated, audit-d5, audit-d6, compare-d5, compare-d6, d5-full, d6-full\n' >&2
    exit 2
    ;;
esac
