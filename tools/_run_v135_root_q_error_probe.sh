#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V135_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

OUT_ROOT="${V135_OUT_ROOT:-/home/laure/alphaxiang/v13_root_q_error_probe}"
AUDIT_ROOT="${V135_AUDIT_ROOT:-/home/laure/alphaxiang/v13_root_decision_audit}"
RANK_ROOT="${V135_RANK_ROOT:-/home/laure/alphaxiang/v13_root_ranking_repair}"
EXPANDED_JSONL="${V135_EXPANDED_JSONL:-$AUDIT_ROOT/root_regret_expanded300_d14d16_top_all.jsonl}"
EXACT_JSONL="${V135_EXACT_JSONL:-$AUDIT_ROOT/root_regret_exact112_d14d16_top_all.jsonl}"
TRAJECTORY_JSON="${V135_TRAJECTORY_JSON:-$RANK_ROOT/trajectory/bad12_full_milestones/bad12_baseline_full.json}"

FEATURE_DIR="$OUT_ROOT/features"
SMOKE_DIR="$OUT_ROOT/smoke"
SIDECAR_DIR="$OUT_ROOT/sidecar"
GATE_DIR="$OUT_ROOT/gate"

DEVICE="${V135_DEVICE:-cuda:0}"
EPOCHS="${V135_EPOCHS:-250}"
BATCH_SIZE="${V135_BATCH_SIZE:-256}"
HIDDEN_DIM="${V135_HIDDEN_DIM:-64}"
LR="${V135_LR:-0.001}"
PHASE="${V135_PHASE:-all}"

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'missing required file: %s\n' "$path" >&2
    exit 2
  fi
}

need_inputs() {
  need_file "$EXPANDED_JSONL"
  need_file "$EXACT_JSONL"
  need_file "$TRAJECTORY_JSON"
}

run_static() {
  "$PY" -m py_compile \
    tools/v13_root_q_error_audit.py \
    tools/v13_root_q_sidecar.py \
    tools/v13_root_q_gate_eval.py
  bash -n tools/_run_v135_root_q_error_probe.sh
}

run_audit_one() {
  local input_jsonl="$1"
  local out_jsonl="$2"
  local out_summary="$3"
  local out_md="$4"
  local max_roots="$5"
  local args=(
    tools/v13_root_q_error_audit.py
    "$input_jsonl"
    --trajectory-json "$TRAJECTORY_JSON"
    --out-jsonl "$out_jsonl"
    --out-summary "$out_summary"
    --out-md "$out_md"
  )
  if [[ "$max_roots" != "0" ]]; then
    args+=(--max-roots "$max_roots")
  fi
  "$PY" "${args[@]}"
}

make_smoke_jsonl() {
  local input_jsonl="$1"
  local out_jsonl="$2"
  local bad_roots="$3"
  local clean_roots="$4"
  "$PY" - "$input_jsonl" "$out_jsonl" "$bad_roots" "$clean_roots" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
bad_target = int(sys.argv[3])
clean_target = int(sys.argv[4])

def root_key(row):
    return "\n".join([
        str(row.get("audit_json", "")),
        str(row.get("fen", "")),
        str(row.get("game_index", "")),
        str(row.get("ply", "")),
        str(row.get("selected_move", "")),
    ])

groups = defaultdict(list)
with src.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            groups[root_key(row)].append(row)

bad = []
clean = []
for key, rows in groups.items():
    selected = next((r for r in rows if r.get("is_selected")), rows[0])
    if float(selected.get("regret_cp", 0.0) or 0.0) >= 150.0:
        bad.append(key)
    else:
        clean.append(key)

chosen = bad[:bad_target] + clean[:clean_target]
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as handle:
    for key in chosen:
        for row in groups[key]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps({"source": str(src), "out": str(dst), "bad_roots": len(bad[:bad_target]), "clean_roots": len(clean[:clean_target]), "rows": sum(len(groups[k]) for k in chosen)}, indent=2), flush=True)
PY
}

run_smoke() {
  need_inputs
  mkdir -p "$SMOKE_DIR"
  make_smoke_jsonl "$EXPANDED_JSONL" "$SMOKE_DIR/expanded_bad2_clean18_root_regret.jsonl" 2 18
  make_smoke_jsonl "$EXACT_JSONL" "$SMOKE_DIR/exact_bad2_clean8_root_regret.jsonl" 2 8
  run_audit_one \
    "$SMOKE_DIR/expanded_bad2_clean18_root_regret.jsonl" \
    "$SMOKE_DIR/expanded20_features.jsonl" \
    "$SMOKE_DIR/expanded20_summary.json" \
    "$SMOKE_DIR/expanded20_summary.md" \
    0
  run_audit_one \
    "$SMOKE_DIR/exact_bad2_clean8_root_regret.jsonl" \
    "$SMOKE_DIR/exact10_features.jsonl" \
    "$SMOKE_DIR/exact10_summary.json" \
    "$SMOKE_DIR/exact10_summary.md" \
    0
  "$PY" tools/v13_root_q_sidecar.py \
    --train-jsonl "$SMOKE_DIR/expanded20_features.jsonl" \
    --holdout-jsonl "$SMOKE_DIR/exact10_features.jsonl" \
    --out-dir "$SMOKE_DIR/sidecar_5epoch" \
    --epochs 5 \
    --batch-size 64 \
    --hidden-dim 32 \
    --lr "$LR" \
    --device "$DEVICE"
  "$PY" tools/v13_root_q_gate_eval.py \
    --sidecar "$SMOKE_DIR/sidecar_5epoch/sidecar.pt" \
    --expanded-jsonl "$SMOKE_DIR/expanded20_features.jsonl" \
    --exact-jsonl "$SMOKE_DIR/exact10_features.jsonl" \
    --out-json "$SMOKE_DIR/gate_smoke.json" \
    --out-md "$SMOKE_DIR/gate_smoke.md" \
    --threshold-json "$SMOKE_DIR/threshold_smoke.json" \
    --device "$DEVICE"
}

run_audit() {
  need_inputs
  mkdir -p "$FEATURE_DIR"
  run_audit_one \
    "$EXPANDED_JSONL" \
    "$FEATURE_DIR/expanded300_features.jsonl" \
    "$FEATURE_DIR/expanded300_summary.json" \
    "$FEATURE_DIR/expanded300_summary.md" \
    0
  run_audit_one \
    "$EXACT_JSONL" \
    "$FEATURE_DIR/exact112_features.jsonl" \
    "$FEATURE_DIR/exact112_summary.json" \
    "$FEATURE_DIR/exact112_summary.md" \
    0
}

run_train() {
  need_file "$FEATURE_DIR/expanded300_features.jsonl"
  need_file "$FEATURE_DIR/exact112_features.jsonl"
  mkdir -p "$SIDECAR_DIR"
  "$PY" tools/v13_root_q_sidecar.py \
    --train-jsonl "$FEATURE_DIR/expanded300_features.jsonl" \
    --holdout-jsonl "$FEATURE_DIR/exact112_features.jsonl" \
    --out-dir "$SIDECAR_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --lr "$LR" \
    --device "$DEVICE"
}

run_gate() {
  need_file "$SIDECAR_DIR/sidecar.pt"
  need_file "$FEATURE_DIR/expanded300_features.jsonl"
  need_file "$FEATURE_DIR/exact112_features.jsonl"
  mkdir -p "$GATE_DIR"
  "$PY" tools/v13_root_q_gate_eval.py \
    --sidecar "$SIDECAR_DIR/sidecar.pt" \
    --expanded-jsonl "$FEATURE_DIR/expanded300_features.jsonl" \
    --exact-jsonl "$FEATURE_DIR/exact112_features.jsonl" \
    --out-json "$GATE_DIR/offline_gate.json" \
    --out-md "$GATE_DIR/offline_gate.md" \
    --threshold-json "$GATE_DIR/threshold.json" \
    --device "$DEVICE"
  "$PY" - "$GATE_DIR/offline_gate.json" "$OUT_ROOT/arena_next_steps.txt" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
if summary.get("passed"):
    threshold = summary["threshold"]
    lines = [
        "V13.5 offline sidecar gate passed.",
        "",
        "Do not auto-launch arena from this runner. First integrate the gate into external_arena.py, then run a paired d5 black-side arena.",
        "",
        "Suggested integration flags:",
        f"  --root-q-sidecar {summary['sidecar']} --root-q-threshold {threshold:.4f} --root-q-top-k 6 --root-q-override-margin-cp 300",
    ]
else:
    combined = summary["combined"]
    exact = summary.get("exact_holdout")
    lines = [
        "V13.5 offline sidecar gate did not pass acceptance.",
        "",
        f"combined bad prevented: {combined['bad_prevented']}/{combined['bad_roots']}",
        f"combined catastrophic prevented: {combined['catastrophic_prevented']}/{combined['catastrophic_roots']}",
        f"combined clean regressions: {combined['new_non_bad_regressions']}/{combined['clean_roots']} ({combined['new_non_bad_regression_rate_pct']:.2f}%)",
    ]
    if exact is not None:
        lines.append(f"exact holdout clean regressions: {exact['new_non_bad_regressions']}/{exact['clean_roots']} ({exact['new_non_bad_regression_rate_pct']:.2f}%)")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines), flush=True)
PY
}

case "$PHASE" in
  static)
    run_static
    ;;
  smoke)
    run_static
    run_smoke
    ;;
  audit)
    run_static
    run_audit
    ;;
  train)
    run_train
    ;;
  gate)
    run_gate
    ;;
  all)
    run_static
    run_smoke
    run_audit
    run_train
    run_gate
    ;;
  *)
    printf 'unknown V135_PHASE: %s\n' "$PHASE" >&2
    printf 'valid phases: static, smoke, audit, train, gate, all\n' >&2
    exit 2
    ;;
esac
