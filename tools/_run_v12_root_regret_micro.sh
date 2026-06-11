#!/usr/bin/env bash
# V12 root-regret micro repair from V12 PEAK.
#
# This is intentionally narrow: it starts from the V12 PEAK checkpoint, consumes
# the 6400-sim root-regret shard produced by the V12.7 re-audit, and trains only
# the policy-head projections with pairwise teacher-Q / bad-move losses. Arena is
# not launched here; snapshots must pass offline root checks first.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V12_RR_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

CKPT="${V12_RR_CKPT:-/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt}"
ANCHOR_CKPT="${V12_RR_ANCHOR_CKPT:-$CKPT}"
DATA_DIR="${V12_RR_DATA_DIR:-v127_reaudit/v12_peak_root_regret_shard_6400_d8}"
ROOT_JSONL="${V12_RR_ROOT_JSONL:-v127_reaudit/v12_peak_root_regret_d4d5_black_6400_d8_all.jsonl}"
EVAL_JSONL="${V12_RR_EVAL_JSONL:-$ROOT_JSONL}"
SOURCE_AUDIT="${V12_RR_SOURCE_AUDIT:-v127_reaudit/v12_peak_d4d5_black_loss_selected_mcts6400_d8_full80.json}"
HUMAN_DATA="${V12_RR_HUMAN_DATA:-/home/laure/alphaxiang/human_bootstrap_data_elite_wdl}"
OUT_ROOT="${V12_RR_OUT_ROOT:-/home/laure/alphaxiang/v12_root_regret_micro}"
TRAIN_ROOT="${V12_RR_TRAIN_ROOT:-/home/laure/alphaxiang/training_runs/run_021_v12_root_regret_micro_from_peak}"
DEVICE="${V12_RR_DEVICE:-cuda:0}"
PHASE="${V12_RR_PHASE:-smoke}"

EVAL_DIR="$OUT_ROOT/offline_eval"
MCTS_GATE_DIR="$OUT_ROOT/mcts_gate"
SMOKE_TRAIN_DIR="${V12_RR_SMOKE_TRAIN_DIR:-/tmp/v12_root_regret_micro_smoke_train}"

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

require_file "$CKPT"
require_file "$ANCHOR_CKPT"
require_file "$ROOT_JSONL"
require_file "$EVAL_JSONL"
require_file "$SOURCE_AUDIT"
require_file "$DATA_DIR/manifest.json"
require_file "$DATA_DIR/train/shard_00000.pt"
require_dir "$HUMAN_DATA"

checkpoint_step() {
  "$PY" - "$1" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("global_step", 0)))
PY
}

sample_count() {
  "$PY" - "$DATA_DIR/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(max(1, int(manifest["total_samples_written"])))
PY
}

run_static() {
  "$PY" -m py_compile \
    tools/v13_root_decision_audit.py \
    tools/v13_root_repair_offline_eval.py \
    tools/v13_root_shard_read_smoke.py
  bash -n tools/_run_v12_root_regret_micro.sh
}

run_read_smoke() {
  "$PY" -u tools/v13_root_shard_read_smoke.py \
    --shard "$DATA_DIR/train/shard_00000.pt" \
    --checkpoint "$CKPT" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --device "${V12_RR_SMOKE_DEVICE:-cpu}" \
    --batch-size "${V12_RR_SMOKE_BATCH_SIZE:-16}" \
    --disable-bf16
}

run_base_eval() {
  mkdir -p "$EVAL_DIR"
  "$PY" -u tools/v13_root_repair_offline_eval.py "$EVAL_JSONL" \
    --checkpoint "$CKPT" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --out-json "$EVAL_DIR/base_v12_peak.json" \
    --out-md "$EVAL_DIR/base_v12_peak.md" \
    --device "$DEVICE"
}

train_policy_head() {
  local out_dir="$1"
  local train_steps="$2"
  local device="$3"
  local base_step
  local target_steps
  local replay_size

  base_step="$(checkpoint_step "$CKPT")"
  target_steps=$((base_step + train_steps))
  replay_size="${V12_RR_REPLAY_BUFFER_SIZE:-$(sample_count)}"
  mkdir -p "$out_dir"

  "$PY" -u xiangqi_train.py \
    --resume-path "$CKPT" \
    --reset-optimizer-on-resume \
    --reset-selfplay-ingest-state-on-resume \
    --human-data-dir "$HUMAN_DATA" \
    --selfplay-dirs "$DATA_DIR" \
    --output-dir "$out_dir" \
    --device "$device" \
    --foreground \
    --disable-promote-best-on-human-val \
    --disable-selfplay-run-quality-gate \
    --poll-interval-s 0.2 \
    --replay-buffer-size "$replay_size" \
    --bootstrap-human-floor 0.0 \
    --learning-rate "${V12_RR_LR:-2e-6}" \
    --warmup-steps 0 \
    --max-steps "$target_steps" \
    --lr-schedule-max-steps "$target_steps" \
    --save-interval-steps "${V12_RR_SAVE_INTERVAL:-250}" \
    --snapshot-interval-steps "${V12_RR_SNAPSHOT_INTERVAL:-250}" \
    --eval-interval-steps "$target_steps" \
    --log-interval-steps "${V12_RR_LOG_INTERVAL:-25}" \
    --micro-batch-size "${V12_RR_MICRO_BATCH:-64}" \
    --grad-accum-steps "${V12_RR_GRAD_ACCUM:-1}" \
    --samples-per-unit "${V12_RR_SAMPLES_PER_UNIT:-16}" \
    --cpu-sampler-workers "${V12_RR_CPU_SAMPLER_WORKERS:-0}" \
    --wdl-loss-weight 0.0 \
    --policy-loss-weight 0.0 \
    --value-loss-weight 0.0 \
    --wdl-value-consistency-weight 0.0 \
    --policy-oracle-alpha 0.0 \
    --teacher-q-loss-weight 0.0 \
    --teacher-q-pairwise-loss-weight "${V12_RR_PAIRWISE_WEIGHT:-1.0}" \
    --teacher-q-pairwise-use-anchor-reference \
    --teacher-q-pairwise-bad-move-only \
    --teacher-q-pairwise-min-gap-cp "${V12_RR_PAIRWISE_MIN_GAP_CP:-150}" \
    --teacher-q-pairwise-margin-logit "${V12_RR_PAIRWISE_MARGIN_LOGIT:-0.35}" \
    --teacher-q-pairwise-beta "${V12_RR_PAIRWISE_BETA:-1.0}" \
    --bad-move-suppression-loss-weight "${V12_RR_BAD_MOVE_WEIGHT:-0.5}" \
    --bad-move-suppression-min-gap-cp "${V12_RR_BAD_MOVE_MIN_GAP_CP:-150}" \
    --bad-move-suppression-margin-logit "${V12_RR_BAD_MOVE_MARGIN_LOGIT:-0.75}" \
    --bad-move-suppression-beta "${V12_RR_BAD_MOVE_BETA:-2.0}" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --anchor-policy-kl-weight "${V12_RR_ANCHOR_POLICY_KL:-0.05}" \
    --anchor-value-mse-weight 0.0 \
    --train-only-policy-head \
    --seed "${V12_RR_SEED:-127021}"
}

run_train_smoke() {
  train_policy_head "$SMOKE_TRAIN_DIR" "${V12_RR_SMOKE_STEPS:-1}" "${V12_RR_TRAIN_SMOKE_DEVICE:-${V12_RR_SMOKE_DEVICE:-cpu}}"
}

run_train_a() {
  train_policy_head "$TRAIN_ROOT/arm_a_policy_head" "${V12_RR_TRAIN_STEPS:-250}" "$DEVICE"
}

run_eval_a() {
  local ckpt="$TRAIN_ROOT/arm_a_policy_head/latest.pt"
  require_file "$ckpt"
  mkdir -p "$EVAL_DIR"
  "$PY" -u tools/v13_root_repair_offline_eval.py "$EVAL_JSONL" \
    --checkpoint "$ckpt" \
    --anchor-checkpoint "$ANCHOR_CKPT" \
    --out-json "$EVAL_DIR/arm_a_policy_head_latest.json" \
    --out-md "$EVAL_DIR/arm_a_policy_head_latest.md" \
    --device "$DEVICE"
}

run_mcts_gate_for_checkpoint() {
  local label="$1"
  local ckpt="$2"
  require_file "$ckpt"
  mkdir -p "$MCTS_GATE_DIR"
  "$PY" -u tools/v13_root_decision_audit.py "$SOURCE_AUDIT" \
    --input-kind audit \
    --checkpoint "$ckpt" \
    --out-json "$MCTS_GATE_DIR/${label}_selected_mcts6400_d8.json" \
    --out-md "$MCTS_GATE_DIR/${label}_selected_mcts6400_d8.md" \
    --only-side black \
    --max-positions "${V12_RR_MCTS_GATE_MAX_POSITIONS:-80}" \
    --ply-stride 1 \
    --selected-source mcts \
    --mcts-sims 6400 \
    --mcts-c-puct "${V12_RR_MCTS_C_PUCT:-1.25}" \
    --mcts-q-weight "${V12_RR_MCTS_Q_WEIGHT:-1.0}" \
    --mcts-q-clip 1.0 \
    --mcts-temperature-move "${V12_RR_MCTS_TEMPERATURE:-0.1}" \
    --mcts-seed "${V12_RR_MCTS_SEED:-20260524}" \
    --pika-root-depth "${V12_RR_PIKA_ROOT_DEPTH:-8}" \
    --pika-child-depth "${V12_RR_PIKA_CHILD_DEPTH:-8}" \
    --pika-root-multipv "${V12_RR_PIKA_ROOT_MULTIPV:-6}" \
    --pika-workers "${V12_RR_PIKA_WORKERS:-8}" \
    --pika-threads-per-worker "${V12_RR_PIKA_THREADS_PER_WORKER:-1}" \
    --pika-hash-mb "${V12_RR_PIKA_HASH_MB:-128}" \
    --device "$DEVICE"
}

run_mcts_gate_base() {
  run_mcts_gate_for_checkpoint "base_v12_peak" "$CKPT"
}

run_mcts_gate_a() {
  local ckpt="$TRAIN_ROOT/arm_a_policy_head/latest.pt"
  run_mcts_gate_for_checkpoint "arm_a_policy_head_latest" "$ckpt"
}

run_mcts_gate_compare() {
  "$PY" - "$MCTS_GATE_DIR/base_v12_peak_selected_mcts6400_d8.json" \
    "$MCTS_GATE_DIR/arm_a_policy_head_latest_selected_mcts6400_d8.json" \
    "$MCTS_GATE_DIR/gate_compare.json" <<'PY'
import json
import sys
from pathlib import Path

base_path, cand_path, out_path = map(Path, sys.argv[1:4])
base = json.loads(base_path.read_text(encoding="utf-8"))["summary"]
cand = json.loads(cand_path.read_text(encoding="utf-8"))["summary"]

def counts(summary):
    row = summary.get("counts", {})
    return {
        "bad_root": int(row.get("bad_root", 0)),
        "catastrophic": int(row.get("catastrophic", 0)),
        "q_inversion": int(row.get("q_inversion", 0)),
        "missing_candidate": int(row.get("missing_candidate", 0)),
    }

base_counts = counts(base)
cand_counts = counts(cand)
passed = (
    cand_counts["bad_root"] < base_counts["bad_root"]
    and cand_counts["catastrophic"] <= base_counts["catastrophic"]
    and cand_counts["missing_candidate"] <= base_counts["missing_candidate"]
)
payload = {
    "passed": passed,
    "gate": "candidate bad_root must improve, catastrophic/missing must not increase",
    "base": base_counts,
    "candidate": cand_counts,
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
raise SystemExit(0 if passed else 1)
PY
}

case "$PHASE" in
  static)
    run_static
    ;;
  read_smoke)
    run_static
    run_read_smoke
    ;;
  train_smoke)
    run_static
    run_read_smoke
    run_train_smoke
    ;;
  base_eval)
    run_base_eval
    ;;
  train_a)
    run_train_a
    ;;
  eval_a)
    run_eval_a
    ;;
  mcts_gate_base)
    run_mcts_gate_base
    ;;
  mcts_gate_a)
    run_mcts_gate_a
    ;;
  mcts_gate_compare)
    run_mcts_gate_compare
    ;;
  smoke)
    run_static
    run_read_smoke
    ;;
  all)
    run_static
    run_read_smoke
    run_base_eval
    run_train_a
    run_eval_a
    run_mcts_gate_base
    run_mcts_gate_a
    run_mcts_gate_compare
    ;;
  *)
    echo "unknown V12_RR_PHASE=$PHASE" >&2
    echo "valid phases: static, read_smoke, train_smoke, base_eval, train_a, eval_a, mcts_gate_base, mcts_gate_a, mcts_gate_compare, smoke, all" >&2
    exit 2
    ;;
esac
