#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V134_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

CKPT="${V134_CKPT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
ANCHOR_CKPT="${V134_ANCHOR_CKPT:-$CKPT}"
AUDIT_ROOT="${V134_AUDIT_ROOT:-/home/laure/alphaxiang/v13_root_decision_audit}"
OUT_ROOT="${V134_OUT_ROOT:-/home/laure/alphaxiang/v13_root_ranking_repair}"
TRAIN_ROOT="${V134_TRAIN_ROOT:-/home/laure/alphaxiang/training_runs/run_033_v134_root_ranking_repair_from031a19000}"
HUMAN_DATA="${V134_HUMAN_DATA:-/home/laure/alphaxiang/human_bootstrap_data_elite_wdl}"
DEVICE="${V134_DEVICE:-cuda:0}"
PHASE="${V134_PHASE:-all}"

EXPANDED_JSONL="$AUDIT_ROOT/root_regret_expanded300_d14d16_top_all.jsonl"
EXACT_JSONL="$AUDIT_ROOT/root_regret_exact112_d14d16_top_all.jsonl"
DATA_DIR="$OUT_ROOT/data/root_regret_selfplay"
SMOKE_DATA_DIR="$OUT_ROOT/smoke/root_regret_selfplay_10"
TRAJ_DIR="$OUT_ROOT/trajectory"
CAL_DIR="$OUT_ROOT/calibration"
EVAL_DIR="$OUT_ROOT/offline_eval"
ARENA_DIR="$OUT_ROOT/arena_search_calibration"
OPENING_SUITE="${V134_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"

mkdir -p "$OUT_ROOT" "$TRAJ_DIR" "$CAL_DIR" "$EVAL_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_file "$CKPT"
require_file "$EXPANDED_JSONL"
require_file "$EXACT_JSONL"

run_static() {
  "$PY" -m py_compile \
    tools/v13_root_trajectory_audit.py \
    tools/v13_root_regret_jsonl_to_shard.py \
    tools/v13_root_repair_offline_eval.py \
    tools/v13_root_shard_read_smoke.py
  bash -n tools/_run_v134_root_ranking_repair.sh
}

run_smoke() {
  mkdir -p "$OUT_ROOT/smoke"
  "$PY" -u tools/v13_root_trajectory_audit.py "$EXPANDED_JSONL" \
    --checkpoint "$CKPT" \
    --out-json "$OUT_ROOT/smoke/trajectory_2roots.json" \
    --out-md "$OUT_ROOT/smoke/trajectory_2roots.md" \
    --position-mode bad \
    --max-roots 2 \
    --milestones 0,32 \
    --search-configs baseline:1.45:1.0 \
    --device "$DEVICE"

  "$PY" -u tools/v13_root_regret_jsonl_to_shard.py "$EXPANDED_JSONL" \
    --exact-jsonl "$EXACT_JSONL" \
    --out-dir "$SMOKE_DATA_DIR" \
    --max-roots 10 \
    --train-fraction 0.8

  "$PY" -u tools/v13_root_shard_read_smoke.py \
    --shard "$SMOKE_DATA_DIR/train/shard_00000.pt" \
    --checkpoint "$CKPT" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --device "$DEVICE" \
    --batch-size 8
}

run_trajectory() {
  "$PY" -u tools/v13_root_trajectory_audit.py "$EXPANDED_JSONL" \
    --checkpoint "$CKPT" \
    --out-json "$TRAJ_DIR/expanded300_baseline_trajectory.json" \
    --out-md "$TRAJ_DIR/expanded300_baseline_trajectory.md" \
    --position-mode all \
    --milestones "${V134_TRAJ_MILESTONES:-0,64,256,1000,3000,8000}" \
    --search-configs baseline:1.45:1.0 \
    --device "$DEVICE"
}

run_calibration() {
  "$PY" -u tools/v13_root_trajectory_audit.py "$EXPANDED_JSONL" \
    --checkpoint "$CKPT" \
    --out-json "$CAL_DIR/bad_control_search_calibration.json" \
    --out-md "$CAL_DIR/bad_control_search_calibration.md" \
    --position-mode bad-control \
    --controls-per-bad 1 \
    --milestones "${V134_CAL_MILESTONES:-8000}" \
    --search-configs "${V134_SEARCH_CONFIGS:-baseline:1.45:1.0,low_cpuct:1.00:1.0,high_q:1.45:1.25,low_cpuct_high_q:1.00:1.25}" \
    --device "$DEVICE"
}

calibration_gate() {
  local path="$CAL_DIR/bad_control_search_calibration.json"
  require_file "$path"
  "$PY" - <<'PY' "$path" "$CAL_DIR/search_gate.json"
import json, sys
payload=json.load(open(sys.argv[1], encoding='utf-8'))
configs=payload.get('summary', {}).get('configs', {})
passing=[]
for name,row in configs.items():
    bad=int(row.get('bad_repaired_known_top', 0))
    judged=int(row.get('bad_judged', 0))
    clean=int(row.get('clean_regressions', 0))
    clean_judged=int(row.get('clean_judged', 0))
    clean_rate=0.0 if clean_judged == 0 else clean/clean_judged
    if judged >= 12 and bad >= 5 and clean_rate <= 0.02:
        passing.append((bad, -clean, -float(row.get('mean_known_top_regret_cp') or 1e9), name, row))
passing.sort(reverse=True)
out={
    'passed': bool(passing),
    'chosen': None if not passing else passing[0][3],
    'chosen_config': None if not passing else passing[0][4],
    'all_configs': configs,
}
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
raise SystemExit(0 if passing else 1)
PY
}

write_command() {
  local run_dir="$1"
  shift
  printf '%q ' "$@" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
}

run_arena_case() {
  local label="$1"
  shift
  local run_dir="$ARENA_DIR/$label"
  mkdir -p "$run_dir"
  local cmd=("$PY" -u tools/external_arena.py "$@" --output-dir "$run_dir")
  write_command "$run_dir" "${cmd[@]}"
  echo "================================================================"
  echo "RUN $label"
  echo "OUT $run_dir"
  echo "================================================================"
  "${cmd[@]}" 2>&1 | tee "$run_dir/runner.log"
}

run_arena_d5_search_calibrated() {
  mkdir -p "$ARENA_DIR"
  local seed="${V134_ARENA_SEED:-2026052201}"
  local openings="${V134_ARENA_OPENINGS:-12}"
  local parallel="${V134_PARALLEL_GAMES:-2}"
  local tuned_cpuct="${V134_TUNED_C_PUCT:-1.0}"
  local tuned_q="${V134_TUNED_Q_WEIGHT:-1.25}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  ARENA_DIR="$ARENA_DIR/d5_${stamp}"
  mkdir -p "$ARENA_DIR"

  common_args=(
    --checkpoint "$CKPT"
    --device "$DEVICE"
    --our-search mcts
    --our-side black
    --opening-suite-path "$OPENING_SUITE"
    --max-openings "$openings"
    --games-per-opening 1
    --parallel-games "$parallel"
    --cross-game-batch-cap 96
    --opp-engine pikafish
    --opp-depth 5
    --opp-threads 1
    --opp-hash-mb 64
    --seed "$seed"
    --our-sims 8000
    --our-q-clip 1.0
    --our-temperature-move 0.02
    --our-root-mate1-blunder-guard
    --our-tactical-mate1-extension
    --our-tactical-mate2-extension
  )

  run_arena_case "baseline_cpuct145_q100" \
    "${common_args[@]}" \
    --our-c-puct 1.45 \
    --our-q-weight 1.0

  run_arena_case "tuned_cpuct${tuned_cpuct}_q${tuned_q}" \
    "${common_args[@]}" \
    --our-c-puct "$tuned_cpuct" \
    --our-q-weight "$tuned_q"
}

run_data() {
  "$PY" -u tools/v13_root_regret_jsonl_to_shard.py "$EXPANDED_JSONL" \
    --exact-jsonl "$EXACT_JSONL" \
    --out-dir "$DATA_DIR" \
    --train-fraction 0.8
}

run_base_eval() {
  "$PY" -u tools/v13_root_repair_offline_eval.py "$EXPANDED_JSONL" "$EXACT_JSONL" \
    --checkpoint "$CKPT" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --out-json "$EVAL_DIR/base_expanded300_plus_exact112.json" \
    --out-md "$EVAL_DIR/base_expanded300_plus_exact112.md" \
    --device "$DEVICE"
}

train_arm_a() {
  run_data
  local base_step
  base_step="$("$PY" - <<'PY' "$CKPT"
import sys, torch
state=torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(int(state.get('global_step', 0)))
PY
)"
  local train_steps="${V134_TRAIN_STEPS:-1000}"
  local target_steps=$((base_step + train_steps))
  local sample_count
  sample_count="$("$PY" - <<'PY' "$DATA_DIR/manifest.json"
import json, sys
print(max(1, int(json.load(open(sys.argv[1]))["total_samples_written"])))
PY
)"
  local replay_size="${V134_REPLAY_BUFFER_SIZE:-$sample_count}"
  mkdir -p "$TRAIN_ROOT/arm_a_policy_head"
  "$PY" -u xiangqi_train.py \
    --model-preset v13_200m_dense \
    --resume-path "$CKPT" \
    --reset-optimizer-on-resume \
    --reset-selfplay-ingest-state-on-resume \
    --human-data-dir "$HUMAN_DATA" \
    --selfplay-dirs "$DATA_DIR" \
    --output-dir "$TRAIN_ROOT/arm_a_policy_head" \
    --device "$DEVICE" \
    --foreground \
    --disable-promote-best-on-human-val \
    --disable-selfplay-run-quality-gate \
    --poll-interval-s 0.2 \
    --replay-buffer-size "$replay_size" \
    --bootstrap-human-floor 0.0 \
    --learning-rate "${V134_LR:-2e-6}" \
    --warmup-steps 0 \
    --max-steps "$target_steps" \
    --lr-schedule-max-steps "$target_steps" \
    --save-interval-steps 250 \
    --snapshot-interval-steps 250 \
    --eval-interval-steps "$target_steps" \
    --log-interval-steps 25 \
    --micro-batch-size "${V134_MICRO_BATCH:-64}" \
    --grad-accum-steps 1 \
    --samples-per-unit 16 \
    --cpu-sampler-workers 0 \
    --wdl-loss-weight 0.0 \
    --policy-loss-weight 0.0 \
    --value-loss-weight 0.0 \
    --wdl-value-consistency-weight 0.0 \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight 0.0 \
    --teacher-q-pairwise-loss-weight 1.0 \
    --teacher-q-pairwise-use-anchor-reference \
    --teacher-q-pairwise-bad-move-only \
    --teacher-q-pairwise-min-gap-cp 150 \
    --teacher-q-pairwise-margin-logit 0.35 \
    --teacher-q-pairwise-beta 1.0 \
    --bad-move-suppression-loss-weight 0.5 \
    --bad-move-suppression-min-gap-cp 150 \
    --bad-move-suppression-margin-logit 0.75 \
    --bad-move-suppression-beta 2.0 \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --anchor-policy-kl-weight 0.05 \
    --anchor-value-mse-weight 0.0 \
    --train-only-policy-head \
    --seed 134202
}

eval_arm_a() {
  local ckpt="$TRAIN_ROOT/arm_a_policy_head/latest.pt"
  require_file "$ckpt"
  "$PY" -u tools/v13_root_repair_offline_eval.py "$EXPANDED_JSONL" "$EXACT_JSONL" \
    --checkpoint "$ckpt" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --out-json "$EVAL_DIR/arm_a_policy_head_latest.json" \
    --out-md "$EVAL_DIR/arm_a_policy_head_latest.md" \
    --device "$DEVICE"
}

case "$PHASE" in
  static)
    run_static
    ;;
  smoke)
    run_static
    run_smoke
    ;;
  trajectory)
    run_trajectory
    ;;
  calibration)
    run_calibration
    ;;
  search_gate)
    calibration_gate
    ;;
  arena_d5)
    run_arena_d5_search_calibrated
    ;;
  data)
    run_data
    ;;
  base_eval)
    run_base_eval
    ;;
  train_a)
    train_arm_a
    ;;
  eval_a)
    eval_arm_a
    ;;
  all)
    run_static
    run_smoke
    run_calibration
    if calibration_gate; then
      echo "search calibration passed; running paired d5 arena and skipping tiny repair training"
      run_arena_d5_search_calibrated
    else
      echo "search calibration failed; falling back to root-regret tiny repair"
      run_data
      run_base_eval
      train_arm_a
      eval_arm_a
    fi
    ;;
  *)
    echo "unknown V134_PHASE=$PHASE" >&2
    exit 2
    ;;
esac
