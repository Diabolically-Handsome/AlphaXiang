#!/usr/bin/env bash
# V12.8 FullPika d20/d20 root-regret data scale-up.
#
# This runner is data-first.  It collects arena sources, labels root positions
# with Pikafish root d20 + child d20, exports JSONL/shards, and refuses to treat
# the output as trainable until the formal d20 root pool reaches the configured
# target size.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V128_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

OUT_ROOT="${V128_OUT_ROOT:-/home/laure/alphaxiang/v128_fullpika_root_retune}"
SCALE_ROOT="${V128_SCALE_ROOT:-$OUT_ROOT/d20_root_scaleup}"
BATCH_ID="${V128_SCALE_BATCH_ID:-$(date -u +%Y%m%d_%H%M%S)}"
BATCH_ROOT="$SCALE_ROOT/batches/$BATCH_ID"
ARENA_ROOT="$BATCH_ROOT/arena"
AUDIT_ROOT="$BATCH_ROOT/audit"
DATA_ROOT="$BATCH_ROOT/data/root_regret"
LOG_ROOT="$BATCH_ROOT/logs"
PHASE="${V128_SCALE_PHASE:-inventory}"
POOL_ROLE="${V128_SCALE_POOL_ROLE:-train_stage1}"

V12_PEAK="${V128_V12_PEAK:-/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt}"
PROBE_B="${V128_PROBE_B:-/home/laure/alphaxiang/training_runs/run_022b_v128_fullpika_root_retune_conservative_from_peak/probe_a_full_model/latest.pt}"
GEN_CKPT="${V128_SCALE_GENERATOR_CKPT:-$PROBE_B}"
if [[ ! -f "$GEN_CKPT" ]]; then
  GEN_CKPT="$V12_PEAK"
fi

OPENING_SUITE="${V128_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"
DEVICE="${V128_DEVICE:-cuda:0}"
TARGET_ROOTS="${V128_SCALE_TARGET_ROOTS:-10000}"

PIKA_ROOT_DEPTH="${V128_PIKA_ROOT_DEPTH:-20}"
PIKA_CHILD_DEPTH="${V128_PIKA_CHILD_DEPTH:-20}"
PIKA_ROOT_MULTIPV="${V128_PIKA_ROOT_MULTIPV:-8}"
PIKA_WORKERS="${V128_PIKA_WORKERS:-16}"
PIKA_THREADS="${V128_PIKA_THREADS_PER_WORKER:-2}"
PIKA_HASH_MB="${V128_PIKA_HASH_MB:-256}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_fullpika_depths() {
  if (( PIKA_ROOT_DEPTH < 20 || PIKA_CHILD_DEPTH < 20 )); then
    echo "refusing non-formal labels: root_depth=$PIKA_ROOT_DEPTH child_depth=$PIKA_CHILD_DEPTH; required d20/d20" >&2
    exit 1
  fi
}

latest_json() {
  local dir="$1"
  local path
  path="$(find "$dir" -maxdepth 1 -type f -name 'external_arena_*.json' | sort | tail -n 1)"
  if [[ -z "$path" ]]; then
    echo "missing arena JSON under $dir" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

audit_summary_value() {
  "$PY" - "$1" "$2" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload.get("summary", {}).get(sys.argv[2], ""))
PY
}

assert_audit_is_d20() {
  local audit_json="$1"
  require_file "$audit_json"
  local root_depth child_depth
  root_depth="$(audit_summary_value "$audit_json" pika_root_depth)"
  child_depth="$(audit_summary_value "$audit_json" pika_child_depth)"
  if [[ -z "$root_depth" || -z "$child_depth" ]]; then
    echo "audit is missing depth metadata: $audit_json" >&2
    exit 1
  fi
  if (( root_depth < 20 || child_depth < 20 )); then
    echo "refusing non-d20 audit: $audit_json root_depth=$root_depth child_depth=$child_depth" >&2
    exit 1
  fi
}

write_status() {
  mkdir -p "$BATCH_ROOT"
  "$PY" - "$BATCH_ROOT/status.json" "$BATCH_ID" "$GEN_CKPT" "$PHASE" "$POOL_ROLE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
status = {}
if path.exists():
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        status = {}
status.update({
    "batch_id": sys.argv[2],
    "generator_checkpoint": sys.argv[3],
    "last_phase": sys.argv[4],
    "pool_role": sys.argv[5],
    "updated_at": datetime.now(timezone.utc).isoformat(),
})
path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

run_static() {
  "$PY" -m py_compile \
    tools/v128_d20_root_data_inventory.py \
    tools/v13_root_decision_audit.py \
    tools/v13_root_regret_export.py \
    tools/v13_root_regret_jsonl_to_shard.py \
    tools/v13_root_shard_read_smoke.py
  bash -n tools/_run_v128_d20_root_scaleup.sh
}

run_inventory() {
  mkdir -p "$SCALE_ROOT"
  "$PY" -u tools/v128_d20_root_data_inventory.py \
    --root "$OUT_ROOT" \
    --target-roots "$TARGET_ROOTS" \
    --out-json "$SCALE_ROOT/inventory_latest.json" \
    --out-md "$SCALE_ROOT/inventory_latest.md"
}

run_collect_depth() {
  local depth="$1"
  require_file "$GEN_CKPT"
  require_file "$OPENING_SUITE"
  mkdir -p "$ARENA_ROOT/d${depth}" "$LOG_ROOT"
  "$PY" -u tools/external_arena.py \
    --checkpoint "$GEN_CKPT" \
    --device "$DEVICE" \
    --our-side black \
    --opening-suite-path "$OPENING_SUITE" \
    --max-openings "${V128_SCALE_MAX_OPENINGS:-12}" \
    --games-per-opening "${V128_SCALE_GAMES_PER_OPENING:-2}" \
    --games "${V128_SCALE_GAMES:-999}" \
    --parallel-games "${V128_SCALE_PARALLEL_GAMES:-2}" \
    --cross-game-batch-cap "${V128_SCALE_CROSS_GAME_BATCH_CAP:-512}" \
    --opp-engine pikafish \
    --opp-depth "$depth" \
    --opp-threads "${V128_SCALE_OPP_THREADS:-1}" \
    --opp-hash-mb "${V128_SCALE_OPP_HASH_MB:-64}" \
    --seed "${V128_SCALE_SEED:-2026060101}" \
    --our-sims "${V128_SCALE_SIMS:-6400}" \
    --our-c-puct "${V128_SCALE_C_PUCT:-1.25}" \
    --our-q-weight "${V128_SCALE_Q_WEIGHT:-1.0}" \
    --our-q-clip 1.0 \
    --our-temperature-move "${V128_SCALE_TEMPERATURE:-0.1}" \
    --our-log-root-stats-top-k "${V128_SCALE_LOG_ROOT_STATS_TOP_K:-16}" \
    --output-dir "$ARENA_ROOT/d${depth}" 2>&1 | tee "$LOG_ROOT/collect_d${depth}.log"
}

run_collect() {
  mkdir -p "$ARENA_ROOT" "$LOG_ROOT"
  for depth in ${V128_SCALE_DEPTHS:-6 7}; do
    run_collect_depth "$depth"
  done
  write_status
}

run_audit() {
  require_fullpika_depths
  mkdir -p "$AUDIT_ROOT" "$LOG_ROOT"
  local inputs=()
  if [[ -n "${V128_SCALE_ARENA_JSONS:-}" ]]; then
    # shellcheck disable=SC2206
    inputs=(${V128_SCALE_ARENA_JSONS})
  else
    for depth in ${V128_SCALE_DEPTHS:-6 7}; do
      inputs+=("$(latest_json "$ARENA_ROOT/d${depth}")")
    done
  fi
  "$PY" -u tools/v13_root_decision_audit.py "${inputs[@]}" \
    --checkpoint "$GEN_CKPT" \
    --out-json "$AUDIT_ROOT/root_decision_audit_${BATCH_ID}_d20d20.json" \
    --out-md "$AUDIT_ROOT/root_decision_audit_${BATCH_ID}_d20d20.md" \
    --opening-suite-path "$OPENING_SUITE" \
    --results "${V128_SCALE_AUDIT_RESULTS:-opp_win,draw,our_win}" \
    --only-side black \
    --max-positions "${V128_SCALE_AUDIT_MAX_POSITIONS:-1000}" \
    --max-positions-per-file "${V128_SCALE_AUDIT_MAX_POSITIONS_PER_FILE:-500}" \
    --ply-stride "${V128_SCALE_AUDIT_PLY_STRIDE:-1}" \
    --selected-source mcts \
    --mcts-sims "${V128_SCALE_AUDIT_MCTS_SIMS:-6400}" \
    --mcts-c-puct "${V128_SCALE_C_PUCT:-1.25}" \
    --mcts-q-weight "${V128_SCALE_Q_WEIGHT:-1.0}" \
    --mcts-q-clip 1.0 \
    --mcts-temperature-move "${V128_SCALE_TEMPERATURE:-0.1}" \
    --pika-root-depth "$PIKA_ROOT_DEPTH" \
    --pika-child-depth "$PIKA_CHILD_DEPTH" \
    --pika-root-multipv "$PIKA_ROOT_MULTIPV" \
    --pika-workers "$PIKA_WORKERS" \
    --pika-threads-per-worker "$PIKA_THREADS" \
    --pika-hash-mb "$PIKA_HASH_MB" \
    --device "$DEVICE" 2>&1 | tee "$LOG_ROOT/audit.log"
  write_status
}

run_export() {
  local audit_json="$AUDIT_ROOT/root_decision_audit_${BATCH_ID}_d20d20.json"
  assert_audit_is_d20 "$audit_json"
  mkdir -p "$DATA_ROOT" "$LOG_ROOT"
  "$PY" -u tools/v13_root_regret_export.py "$audit_json" \
    --out-jsonl "$DATA_ROOT/root_regret_${BATCH_ID}_all.jsonl" \
    --out-summary "$DATA_ROOT/root_regret_${BATCH_ID}_all_summary.json"
  "$PY" -u tools/v13_root_regret_export.py "$audit_json" \
    --only-selected-or-refuted \
    --out-jsonl "$DATA_ROOT/root_regret_${BATCH_ID}_selected_or_refuted.jsonl" \
    --out-summary "$DATA_ROOT/root_regret_${BATCH_ID}_selected_or_refuted_summary.json"
  "$PY" -u tools/v13_root_regret_jsonl_to_shard.py "$DATA_ROOT/root_regret_${BATCH_ID}_all.jsonl" \
    --out-dir "$DATA_ROOT/shard" \
    --train-fraction "${V128_SCALE_TRAIN_FRACTION:-0.85}" \
    --seed "${V128_SCALE_SEED:-2026060101}"
  "$PY" - "$DATA_ROOT/shard/manifest.json" "$audit_json" "$TARGET_ROOTS" <<'PY'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
target_roots = int(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
summary = audit.get("summary", {})
manifest["label_status"] = "fullpika_d20d20"
manifest["fullpika_ok"] = True
manifest["training_allowed_before_global_target"] = False
manifest["global_target_roots"] = target_roots
manifest["fullpika_depths"] = {
    "pika_root_depth": summary.get("pika_root_depth"),
    "pika_child_depth": summary.get("pika_child_depth"),
    "pika_root_multipv": summary.get("pika_root_multipv"),
    "mcts_sims": summary.get("mcts_sims"),
}
manifest["audit_json"] = str(audit_path)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  "$PY" - "$DATA_ROOT/shard/manifest.json" "$POOL_ROLE" <<'PY'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
role = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["pool_role"] = role
manifest["validation_holdout"] = role.startswith("val_")
manifest["training_allowed"] = False if role.startswith("val_") else bool(manifest.get("training_allowed_before_global_target", False))
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  local inventory_json="$LOG_ROOT/inventory_after_export.json"
  "$PY" -u tools/v128_d20_root_data_inventory.py \
    --root "$OUT_ROOT" \
    --target-roots "$TARGET_ROOTS" \
    --out-json "$inventory_json" \
    --out-md "$LOG_ROOT/inventory_after_export.md" >/dev/null
  "$PY" - "$DATA_ROOT/shard/manifest.json" "$inventory_json" "$POOL_ROLE" <<'PY'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
inventory_path = Path(sys.argv[2])
role = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
training_allowed = bool(inventory.get("training_allowed")) and not role.startswith("val_")
manifest["training_allowed_before_global_target"] = bool(inventory.get("training_allowed"))
manifest["training_allowed"] = training_allowed
manifest["global_formal_unique_roots_at_export"] = (
    inventory.get("formal_d20d20", {}) or {}
).get("unique_roots")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  if "$PY" - "$DATA_ROOT/shard/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if manifest.get("training_allowed") else 1)
PY
  then
    find "$DATA_ROOT/shard" -maxdepth 1 -type f -name 'DO_NOT_TRAIN*' -delete
  else
    printf '%s\n' "This shard is formal d20/d20, but global root pool is below target or this is a validation holdout. Do not train." \
      > "$DATA_ROOT/shard/DO_NOT_TRAIN_until_${TARGET_ROOTS}_formal_roots.txt"
  fi
  write_status
}

run_read_smoke() {
  require_file "$DATA_ROOT/shard/train/shard_00000.pt"
  "$PY" -u tools/v13_root_shard_read_smoke.py \
    --shard "$DATA_ROOT/shard/train/shard_00000.pt" \
    --checkpoint "$GEN_CKPT" \
    --anchor-checkpoint "$V12_PEAK" \
    --device "${V128_SCALE_SMOKE_DEVICE:-cpu}" \
    --batch-size "${V128_SCALE_SMOKE_BATCH_SIZE:-32}" \
    --disable-bf16
}

case "$PHASE" in
  static)
    run_static
    ;;
  inventory)
    run_inventory
    ;;
  collect)
    run_collect
    ;;
  audit)
    run_audit
    ;;
  export)
    run_export
    ;;
  read_smoke)
    run_read_smoke
    ;;
  batch)
    run_static
    run_collect
    run_audit
    run_export
    run_read_smoke
    run_inventory
    ;;
  *)
    echo "unknown V128_SCALE_PHASE=$PHASE" >&2
    echo "valid phases: static, inventory, collect, audit, export, read_smoke, batch" >&2
    exit 2
    ;;
esac
