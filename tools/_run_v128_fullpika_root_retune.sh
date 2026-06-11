#!/usr/bin/env bash
# V12.8 FullPika Root-Retune.
#
# Clean retune from V12 PEAK.  The key difference from the old V12.7 curriculum:
# root-regret/FullPika shards are generated and fully labeled before training.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V128_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

V12_PEAK="${V128_V12_PEAK:-/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt}"
OUT_ROOT="${V128_OUT_ROOT:-/home/laure/alphaxiang/v128_fullpika_root_retune}"
TRAIN_ROOT="${V128_TRAIN_ROOT:-/home/laure/alphaxiang/training_runs/run_022_v128_fullpika_root_retune_from_peak}"
HUMAN_DATA="${V128_HUMAN_DATA:-/home/laure/alphaxiang/human_bootstrap_data_elite_wdl}"
OPENING_SUITE="${V128_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"
DEVICE="${V128_DEVICE:-cuda:0}"
PHASE="${V128_PHASE:-static}"

ARENA_ROOT="$OUT_ROOT/arena"
AUDIT_ROOT="$OUT_ROOT/audit"
DATA_ROOT="$OUT_ROOT/data/root_regret"
EVAL_ROOT="$OUT_ROOT/offline_eval"
SMOKE_TRAIN_ROOT="${V128_SMOKE_TRAIN_ROOT:-/tmp/v128_fullpika_root_retune_smoke}"
IMPORT_JSONL="${V128_IMPORT_JSONL:-v127_reaudit/v12_peak_root_regret_d4d5_black_6400_d8_all.jsonl}"

PIKA_WORKERS="${V128_PIKA_WORKERS:-16}"
PIKA_THREADS="${V128_PIKA_THREADS_PER_WORKER:-2}"
PIKA_HASH_MB="${V128_PIKA_HASH_MB:-256}"
FULLPIKA_MIN_ROOT_DEPTH=20
FULLPIKA_MIN_CHILD_DEPTH=20

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$V12_PEAK"
require_file "$OPENING_SUITE"
require_dir "$HUMAN_DATA"
mkdir -p "$OUT_ROOT" "$ARENA_ROOT" "$AUDIT_ROOT" "$DATA_ROOT" "$EVAL_ROOT"

checkpoint_step() {
  "$PY" - "$1" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("global_step", 0)))
PY
}

sample_count() {
  "$PY" - "$DATA_ROOT/shard/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(max(1, int(manifest["total_samples_written"])))
PY
}

audit_summary_value() {
  "$PY" - "$AUDIT_ROOT/root_decision_audit_d5d6.json" "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("summary", {}).get(sys.argv[2], ""))
PY
}

assert_fullpika_audit() {
  require_file "$AUDIT_ROOT/root_decision_audit_d5d6.json"
  local root_depth child_depth
  root_depth="$(audit_summary_value pika_root_depth)"
  child_depth="$(audit_summary_value pika_child_depth)"
  if [[ -z "$root_depth" || -z "$child_depth" ]]; then
    echo "audit is missing Pika depth metadata: $AUDIT_ROOT/root_decision_audit_d5d6.json" >&2
    exit 1
  fi
  if (( root_depth < FULLPIKA_MIN_ROOT_DEPTH || child_depth < FULLPIKA_MIN_CHILD_DEPTH )); then
    echo "refusing to train/export non-FullPika labels: root_depth=$root_depth child_depth=$child_depth required_root_depth>=$FULLPIKA_MIN_ROOT_DEPTH required_child_depth>=$FULLPIKA_MIN_CHILD_DEPTH" >&2
    echo "rerun V128_PHASE=audit with V128_PIKA_ROOT_DEPTH>=$FULLPIKA_MIN_ROOT_DEPTH and V128_PIKA_CHILD_DEPTH>=$FULLPIKA_MIN_CHILD_DEPTH" >&2
    exit 1
  fi
}

run_static() {
  "$PY" -m py_compile \
    tools/v13_root_decision_audit.py \
    tools/v13_root_regret_export.py \
    tools/v13_root_regret_jsonl_to_shard.py \
    tools/v13_root_repair_offline_eval.py \
    tools/v13_root_shard_read_smoke.py \
    tools/root_topk_verifier_offline_eval.py
  bash -n tools/_run_v128_fullpika_root_retune.sh
}

run_arena_collect() {
  local depth="$1"
  local label="d${depth}"
  local out_dir="$ARENA_ROOT/$label"
  mkdir -p "$out_dir"
  "$PY" -u tools/external_arena.py \
    --checkpoint "$V12_PEAK" \
    --device "$DEVICE" \
    --our-side black \
    --opening-suite-path "$OPENING_SUITE" \
    --max-openings "${V128_COLLECT_OPENINGS:-12}" \
    --games-per-opening "${V128_GAMES_PER_OPENING:-1}" \
    --parallel-games "${V128_PARALLEL_GAMES:-1}" \
    --opp-engine pikafish \
    --opp-depth "$depth" \
    --opp-threads "${V128_OPP_THREADS:-1}" \
    --opp-hash-mb "${V128_OPP_HASH_MB:-64}" \
    --seed "${V128_SEED:-2026052801}" \
    --our-sims "${V128_COLLECT_SIMS:-6400}" \
    --our-c-puct "${V128_C_PUCT:-1.25}" \
    --our-q-weight "${V128_Q_WEIGHT:-1.0}" \
    --our-q-clip 1.0 \
    --our-temperature-move "${V128_TEMPERATURE:-0.1}" \
    --output-dir "$out_dir" 2>&1 | tee "$out_dir/run.log"
}

latest_json() {
  local dir="$1"
  local path
  path="$(find "$dir" -maxdepth 1 -name 'external_arena_*.json' -type f | sort | tail -n 1)"
  if [[ -z "$path" ]]; then
    echo "missing arena JSON under $dir" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

run_audit() {
  local inputs=()
  if [[ -d "$ARENA_ROOT/d5" ]]; then
    inputs+=("$(latest_json "$ARENA_ROOT/d5")")
  fi
  if [[ -d "$ARENA_ROOT/d6" ]]; then
    inputs+=("$(latest_json "$ARENA_ROOT/d6")")
  fi
  if [[ "${#inputs[@]}" -eq 0 ]]; then
    echo "no collected d5/d6 arena JSONs found under $ARENA_ROOT" >&2
    exit 1
  fi
  "$PY" -u tools/v13_root_decision_audit.py "${inputs[@]}" \
    --checkpoint "$V12_PEAK" \
    --out-json "$AUDIT_ROOT/root_decision_audit_d5d6.json" \
    --out-md "$AUDIT_ROOT/root_decision_audit_d5d6.md" \
    --opening-suite-path "$OPENING_SUITE" \
    --results "${V128_AUDIT_RESULTS:-opp_win,draw,our_win}" \
    --only-side black \
    --max-positions "${V128_AUDIT_MAX_POSITIONS:-300}" \
    --max-positions-per-file "${V128_AUDIT_MAX_POSITIONS_PER_FILE:-150}" \
    --ply-stride "${V128_AUDIT_PLY_STRIDE:-2}" \
    --selected-source mcts \
    --mcts-sims "${V128_AUDIT_MCTS_SIMS:-6400}" \
    --mcts-c-puct "${V128_C_PUCT:-1.25}" \
    --mcts-q-weight "${V128_Q_WEIGHT:-1.0}" \
    --mcts-q-clip 1.0 \
    --mcts-temperature-move "${V128_TEMPERATURE:-0.1}" \
    --pika-root-depth "${V128_PIKA_ROOT_DEPTH:-20}" \
    --pika-child-depth "${V128_PIKA_CHILD_DEPTH:-20}" \
    --pika-root-multipv "${V128_PIKA_ROOT_MULTIPV:-8}" \
    --pika-workers "$PIKA_WORKERS" \
    --pika-threads-per-worker "$PIKA_THREADS" \
    --pika-hash-mb "$PIKA_HASH_MB" \
    --device "$DEVICE"
}

run_export() {
  require_file "$AUDIT_ROOT/root_decision_audit_d5d6.json"
  assert_fullpika_audit
  "$PY" -u tools/v13_root_regret_export.py "$AUDIT_ROOT/root_decision_audit_d5d6.json" \
    --out-jsonl "$DATA_ROOT/root_regret_d5d6_all.jsonl" \
    --out-summary "$DATA_ROOT/root_regret_d5d6_all_summary.json"
  "$PY" -u tools/v13_root_regret_export.py "$AUDIT_ROOT/root_decision_audit_d5d6.json" \
    --only-selected-or-refuted \
    --out-jsonl "$DATA_ROOT/root_regret_d5d6_selected_or_refuted.jsonl" \
    --out-summary "$DATA_ROOT/root_regret_d5d6_selected_or_refuted_summary.json"
  "$PY" -u tools/v13_root_regret_jsonl_to_shard.py "$DATA_ROOT/root_regret_d5d6_all.jsonl" \
    --out-dir "$DATA_ROOT/shard" \
    --train-fraction "${V128_TRAIN_FRACTION:-0.85}" \
    --seed "${V128_SEED:-2026052801}"
  "$PY" - "$DATA_ROOT/shard/manifest.json" "$AUDIT_ROOT/root_decision_audit_d5d6.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
summary = audit.get("summary", {})
manifest["label_status"] = "fullpika"
manifest["fullpika_ok"] = True
manifest["fullpika_depths"] = {
    "pika_root_depth": summary.get("pika_root_depth"),
    "pika_child_depth": summary.get("pika_child_depth"),
    "pika_root_multipv": summary.get("pika_root_multipv"),
    "mcts_sims": summary.get("mcts_sims"),
}
manifest["audit_json"] = str(audit_path)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_import_existing_smoke_data() {
  require_file "$IMPORT_JSONL"
  mkdir -p "$DATA_ROOT"
  cp "$IMPORT_JSONL" "$DATA_ROOT/root_regret_d5d6_all.jsonl"
  "$PY" -u tools/v13_root_regret_jsonl_to_shard.py "$DATA_ROOT/root_regret_d5d6_all.jsonl" \
    --out-dir "$DATA_ROOT/shard" \
    --train-fraction "${V128_TRAIN_FRACTION:-0.85}" \
    --seed "${V128_SEED:-2026052801}"
}

run_read_smoke() {
  require_file "$DATA_ROOT/shard/train/shard_00000.pt"
  "$PY" -u tools/v13_root_shard_read_smoke.py \
    --shard "$DATA_ROOT/shard/train/shard_00000.pt" \
    --checkpoint "$V12_PEAK" \
    --anchor-checkpoint "$V12_PEAK" \
    --device "${V128_SMOKE_DEVICE:-cpu}" \
    --batch-size "${V128_SMOKE_BATCH_SIZE:-16}" \
    --disable-bf16
}

train_common() {
  local out_dir="$1"
  local train_steps="$2"
  local device="$3"
  require_file "$DATA_ROOT/shard/manifest.json"
  assert_fullpika_audit
  local base_step target_steps replay_size
  base_step="$(checkpoint_step "$V12_PEAK")"
  target_steps=$((base_step + train_steps))
  replay_size="${V128_REPLAY_BUFFER_SIZE:-$(sample_count)}"
  mkdir -p "$out_dir"
  "$PY" -u xiangqi_train.py \
    --resume-path "$V12_PEAK" \
    --reset-optimizer-on-resume \
    --reset-selfplay-ingest-state-on-resume \
    --human-data-dir "$HUMAN_DATA" \
    --selfplay-dirs "$DATA_ROOT/shard" \
    --output-dir "$out_dir" \
    --device "$device" \
    --foreground \
    --disable-promote-best-on-human-val \
    --disable-selfplay-run-quality-gate \
    --poll-interval-s 0.2 \
    --replay-buffer-size "$replay_size" \
    --bootstrap-human-floor "${V128_HUMAN_FLOOR:-0.30}" \
    --learning-rate "${V128_LR:-1e-5}" \
    --warmup-steps 0 \
    --max-steps "$target_steps" \
    --lr-schedule-max-steps "$target_steps" \
    --save-interval-steps "${V128_SAVE_INTERVAL:-250}" \
    --snapshot-interval-steps "${V128_SNAPSHOT_INTERVAL:-250}" \
    --eval-interval-steps "$target_steps" \
    --log-interval-steps "${V128_LOG_INTERVAL:-25}" \
    --micro-batch-size "${V128_MICRO_BATCH:-128}" \
    --grad-accum-steps "${V128_GRAD_ACCUM:-1}" \
    --samples-per-unit "${V128_SAMPLES_PER_UNIT:-16}" \
    --cpu-sampler-workers "${V128_CPU_SAMPLER_WORKERS:-0}" \
    --wdl-loss-weight "${V128_WDL_LOSS_WEIGHT:-0.20}" \
    --policy-loss-weight "${V128_POLICY_LOSS_WEIGHT:-0.30}" \
    --value-loss-weight "${V128_VALUE_LOSS_WEIGHT:-0.10}" \
    --wdl-value-consistency-weight "${V128_WDL_VALUE_CONSISTENCY:-0.00}" \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight "${V128_TEACHER_Q_WEIGHT:-0.10}" \
    --teacher-q-temperature-cp "${V128_TEACHER_Q_TEMP_CP:-80}" \
    --teacher-q-pairwise-loss-weight "${V128_PAIRWISE_WEIGHT:-1.0}" \
    --teacher-q-pairwise-use-anchor-reference \
    --teacher-q-pairwise-min-gap-cp "${V128_PAIRWISE_MIN_GAP_CP:-150}" \
    --teacher-q-pairwise-margin-logit "${V128_PAIRWISE_MARGIN_LOGIT:-0.35}" \
    --teacher-q-pairwise-beta "${V128_PAIRWISE_BETA:-1.0}" \
    --bad-move-suppression-loss-weight "${V128_BAD_MOVE_WEIGHT:-0.25}" \
    --bad-move-suppression-min-gap-cp "${V128_BAD_MOVE_MIN_GAP_CP:-150}" \
    --bad-move-suppression-margin-logit "${V128_BAD_MOVE_MARGIN_LOGIT:-0.75}" \
    --bad-move-suppression-beta "${V128_BAD_MOVE_BETA:-2.0}" \
    --anchor-checkpoint "$V12_PEAK" \
    --anchor-policy-kl-weight "${V128_ANCHOR_POLICY_KL:-0.05}" \
    --anchor-value-mse-weight "${V128_ANCHOR_VALUE_MSE:-0.05}" \
    --seed "${V128_SEED:-2026052801}"
}

run_train_smoke() {
  train_common "$SMOKE_TRAIN_ROOT" "${V128_SMOKE_STEPS:-3}" "${V128_TRAIN_SMOKE_DEVICE:-$DEVICE}"
}

run_train_probe() {
  train_common "$TRAIN_ROOT/probe_a_full_model" "${V128_TRAIN_STEPS:-1000}" "$DEVICE"
}

run_eval_probe() {
  local ckpt="$TRAIN_ROOT/probe_a_full_model/latest.pt"
  require_file "$ckpt"
  require_file "$DATA_ROOT/root_regret_d5d6_all.jsonl"
  "$PY" -u tools/v13_root_repair_offline_eval.py "$DATA_ROOT/root_regret_d5d6_all.jsonl" \
    --checkpoint "$ckpt" \
    --anchor-checkpoint "$V12_PEAK" \
    --out-json "$EVAL_ROOT/probe_a_offline_eval.json" \
    --out-md "$EVAL_ROOT/probe_a_offline_eval.md" \
    --device "$DEVICE"
}

case "$PHASE" in
  static)
    run_static
    ;;
  collect_d5)
    run_arena_collect 5
    ;;
  collect_d6)
    run_arena_collect 6
    ;;
  audit)
    run_audit
    ;;
  export)
    run_export
    ;;
  import_existing_smoke)
    run_import_existing_smoke_data
    ;;
  read_smoke)
    run_read_smoke
    ;;
  smoke_existing)
    run_static
    run_import_existing_smoke_data
    run_read_smoke
    ;;
  train_smoke)
    run_static
    run_read_smoke
    run_train_smoke
    ;;
  train_probe)
    run_train_probe
    ;;
  eval_probe)
    run_eval_probe
    ;;
  *)
    echo "unknown V128_PHASE=$PHASE" >&2
    echo "valid phases: static, collect_d5, collect_d6, audit, export, import_existing_smoke, read_smoke, smoke_existing, train_smoke, train_probe, eval_probe" >&2
    exit 2
    ;;
esac
