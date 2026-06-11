#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"

PY="${V14V_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
CKPT="${V14V_CKPT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
OPENING_SUITE="${V14V_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"
OUT="${V14V_OUT:-/home/laure/alphaxiang/v14v_pikaverifier/phase3_paired_ab_d5_black_top3_d10_m300_danger600_$(date +%Y%m%d_%H%M%S)_setsid}"
DEVICE="${V14V_DEVICE:-cuda:0}"
PARALLEL_GAMES="${V14V_PARALLEL_GAMES:-2}"
SEEDS="${V14V_SEEDS:-2026051601 2026051602}"

mkdir -p "$OUT"

run_arena() {
  local label="$1"
  local seed="$2"
  local run_dir="$OUT/${label}_seed${seed}"
  mkdir -p "$run_dir"

  local cmd=(
    "$PY" -u tools/external_arena.py
    --checkpoint "$CKPT"
    --output-dir "$run_dir"
    --games 1
    --our-side black
    --opening-suite-path "$OPENING_SUITE"
    --games-per-opening 1
    --max-openings 12
    --parallel-games "$PARALLEL_GAMES"
    --cross-game-batch-cap 96
    --device "$DEVICE"
    --seed "$seed"
    --opp-engine pikafish
    --opp-depth 5
    --opp-threads 1
    --opp-hash-mb 64
    --our-sims 8000
    --our-c-puct 1.45
    --our-q-weight 1.0
    --our-q-clip 1.0
    --our-temperature-move 0.02
    --our-root-mate1-blunder-guard
    --our-tactical-mate1-extension
    --our-tactical-mate2-extension
  )

  if [[ "$label" == "verifier" ]]; then
    cmd+=(
      --our-pikafish-verifier
      --our-verifier-top-k 3
      --our-verifier-depth 10
      --our-verifier-margin-cp 300
      --our-verifier-danger-threshold-cp 600
      --our-verifier-side black
    )
  fi

  printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"

  echo "================================================================"
  echo "RUN $label seed=$seed"
  echo "OUT $run_dir"
  echo "================================================================"

  if [[ "${V14V_DRY_RUN:-0}" == "1" ]]; then
    cat "$run_dir/command.txt"
    return 0
  fi

  "${cmd[@]}" 2>&1 | tee "$run_dir/runner.log"
}

for seed in $SEEDS; do
  run_arena "baseline" "$seed"
  run_arena "verifier" "$seed"
done

if [[ "${V14V_DRY_RUN:-0}" != "1" ]]; then
  "$PY" tools/verifier_event_audit.py "$OUT" \
    --out-md "$OUT/verifier_event_audit.md" \
    --out-json "$OUT/verifier_event_audit.json"
fi

echo "paired A/B output: $OUT"
