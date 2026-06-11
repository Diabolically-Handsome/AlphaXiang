#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"

PY="${V13_AB_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

CKPT="${V13_AB_CKPT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
OPENING_SUITE="${V13_AB_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"
OUT_ROOT="${V13_AB_OUT_ROOT:-/home/laure/alphaxiang/v13_ab_search_diagnostic}"
DEVICE="${V13_AB_DEVICE:-cuda:0}"
PARALLEL_GAMES="${V13_AB_PARALLEL_GAMES:-2}"
PHASE="${V13_AB_PHASE:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_ROOT/${PHASE}_${STAMP}"

mkdir -p "$OUT"

write_command() {
  local run_dir="$1"
  shift
  printf '%q ' "$@" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
}

run_arena() {
  local label="$1"
  shift
  local run_dir="$OUT/$label"
  mkdir -p "$run_dir"
  local cmd=("$PY" -u tools/external_arena.py "$@" --output-dir "$run_dir")

  write_command "$run_dir" "${cmd[@]}"
  echo "================================================================"
  echo "RUN $label"
  echo "OUT $run_dir"
  echo "================================================================"

  if [[ "${V13_AB_DRY_RUN:-0}" == "1" ]]; then
    cat "$run_dir/command.txt"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$run_dir/runner.log"
}

common_model_args=(
  --checkpoint "$CKPT"
  --device "$DEVICE"
)

common_mcts_args=(
  --our-search mcts
  --our-sims 8000
  --our-c-puct 1.45
  --our-q-weight 1.0
  --our-q-clip 1.0
  --our-temperature-move 0.02
  --our-root-mate1-blunder-guard
  --our-tactical-mate1-extension
  --our-tactical-mate2-extension
)

common_ab_args=(
  --our-search alphabeta
  --our-sims 8000
  --our-c-puct 1.45
  --our-q-weight 1.0
  --our-q-clip 1.0
  --our-temperature-move 0.02
  --our-root-mate1-blunder-guard
  --our-root-mate2-blunder-guard
  --our-tactical-mate1-extension
  --our-tactical-mate2-extension
)

common_opening_args=(
  --our-side black
  --opening-suite-path "$OPENING_SUITE"
  --games-per-opening 1
  --parallel-games "$PARALLEL_GAMES"
  --cross-game-batch-cap 96
  --opp-engine pikafish
  --opp-threads 1
  --opp-hash-mb 64
)

run_smoke() {
  run_arena "smoke_random_ab_d1" \
    "${common_model_args[@]}" \
    --games 2 \
    --our-side black \
    --parallel-games 1 \
    --opp-random \
    --max-plies "${V13_AB_SMOKE_MAX_PLIES:-80}" \
    "${common_ab_args[@]}" \
    --ab-depth 1 \
    --ab-root-max-branch 16 \
    --ab-max-branch 8 \
    --ab-quiescence-depth 0

  run_arena "smoke_pika_d1_ab_d1" \
    "${common_model_args[@]}" \
    --games 2 \
    --our-side black \
    --parallel-games 1 \
    --opp-engine pikafish \
    --opp-depth 1 \
    --opp-threads 1 \
    --opp-hash-mb 64 \
    --max-plies "${V13_AB_SMOKE_MAX_PLIES:-80}" \
    "${common_ab_args[@]}" \
    --ab-depth 1 \
    --ab-root-max-branch 16 \
    --ab-max-branch 8 \
    --ab-quiescence-depth 0
}

run_micro() {
  local n="${V13_AB_MICRO_OPENINGS:-4}"
  run_arena "micro_d5_mcts_baseline" \
    "${common_model_args[@]}" \
    "${common_opening_args[@]}" \
    --max-openings "$n" \
    --opp-depth 5 \
    "${common_mcts_args[@]}"

  run_arena "micro_d5_ab_d3" \
    "${common_model_args[@]}" \
    "${common_opening_args[@]}" \
    --max-openings "$n" \
    --opp-depth 5 \
    "${common_ab_args[@]}" \
    --ab-depth 3 \
    --ab-root-max-branch 32 \
    --ab-max-branch 16 \
    --ab-quiescence-depth 1 \
    --ab-quiescence-max-branch 8

  run_arena "micro_d5_ab_d4" \
    "${common_model_args[@]}" \
    "${common_opening_args[@]}" \
    --max-openings "$n" \
    --opp-depth 5 \
    "${common_ab_args[@]}" \
    --ab-depth 4 \
    --ab-root-max-branch 32 \
    --ab-max-branch 12 \
    --ab-quiescence-depth 1 \
    --ab-quiescence-max-branch 8

  run_arena "micro_d5_ab_d5" \
    "${common_model_args[@]}" \
    "${common_opening_args[@]}" \
    --max-openings "$n" \
    --opp-depth 5 \
    "${common_ab_args[@]}" \
    --ab-depth 5 \
    --ab-root-max-branch 24 \
    --ab-max-branch 8 \
    --ab-quiescence-depth 1 \
    --ab-quiescence-max-branch 8
}

selected_ab_args() {
  local preset="${V13_AB_PRESET:-d4}"
  case "$preset" in
    d3)
      echo "--ab-depth 3 --ab-root-max-branch 32 --ab-max-branch 16 --ab-quiescence-depth 1 --ab-quiescence-max-branch 8"
      ;;
    d4)
      echo "--ab-depth 4 --ab-root-max-branch 32 --ab-max-branch 12 --ab-quiescence-depth 1 --ab-quiescence-max-branch 8"
      ;;
    d5)
      echo "--ab-depth 5 --ab-root-max-branch 24 --ab-max-branch 8 --ab-quiescence-depth 1 --ab-quiescence-max-branch 8"
      ;;
    *)
      echo "unknown V13_AB_PRESET=$preset; expected d3/d4/d5" >&2
      return 2
      ;;
  esac
}

run_paired_depth() {
  local depth="$1"
  local openings="${V13_AB_PAIRED_OPENINGS:-12}"
  local seeds="${V13_AB_SEEDS:-2026051901}"
  local ab_args
  read -r -a ab_args <<< "$(selected_ab_args)"

  for seed in $seeds; do
    run_arena "paired_d${depth}_seed${seed}_mcts_baseline" \
      "${common_model_args[@]}" \
      "${common_opening_args[@]}" \
      --seed "$seed" \
      --max-openings "$openings" \
      --opp-depth "$depth" \
      "${common_mcts_args[@]}"

    run_arena "paired_d${depth}_seed${seed}_ab_${V13_AB_PRESET:-d4}" \
      "${common_model_args[@]}" \
      "${common_opening_args[@]}" \
      --seed "$seed" \
      --max-openings "$openings" \
      --opp-depth "$depth" \
      "${common_ab_args[@]}" \
      "${ab_args[@]}"
  done
}

case "$PHASE" in
  smoke)
    run_smoke
    ;;
  micro)
    run_micro
    ;;
  paired_d5)
    run_paired_depth 5
    ;;
  paired_d6)
    run_paired_depth 6
    ;;
  all)
    run_smoke
    run_micro
    run_paired_depth 5
    if [[ "${V13_AB_RUN_D6:-0}" == "1" ]]; then
      run_paired_depth 6
    else
      echo "Set V13_AB_RUN_D6=1 to run the paired d6 block after reviewing d5."
    fi
    ;;
  *)
    echo "unknown V13_AB_PHASE=$PHASE; expected smoke/micro/paired_d5/paired_d6/all" >&2
    exit 2
    ;;
esac

echo "V13 alpha-beta diagnostic output: $OUT"
