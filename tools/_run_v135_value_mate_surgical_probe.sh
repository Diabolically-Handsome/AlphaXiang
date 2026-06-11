#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V135_SURGICAL_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

OUT_ROOT="${V135_SURGICAL_OUT_ROOT:-/home/laure/alphaxiang/v13_value_mate_surgical_probe}"
AUDIT_ROOT="${V135_AUDIT_ROOT:-/home/laure/alphaxiang/v13_root_decision_audit}"
RANK_ROOT="${V135_RANK_ROOT:-/home/laure/alphaxiang/v13_root_ranking_repair}"
CHECKPOINT="${V135_CHECKPOINT:-/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt}"
EXPANDED="${V135_EXPANDED_JSONL:-$AUDIT_ROOT/root_regret_expanded300_d14d16_top_all.jsonl}"
EXACT="${V135_EXACT_JSONL:-$AUDIT_ROOT/root_regret_exact112_d14d16_top_all.jsonl}"
TRAJ="${V135_TRAJECTORY_JSON:-$RANK_ROOT/trajectory/bad12_full_milestones/bad12_baseline_full.json}"
DEVICE="${V135_DEVICE:-cuda:0}"
PHASE="${V135_PHASE:-all}"

need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'missing required file: %s\n' "$path" >&2
    exit 2
  fi
}

need_inputs() {
  need_file "$CHECKPOINT"
  need_file "$EXPANDED"
  need_file "$EXACT"
  need_file "$TRAJ"
}

run_static() {
  "$PY" -m py_compile tools/v13_root_trajectory_audit.py tools/v13_root_mate_veto_eval.py
  bash -n tools/_run_v135_value_mate_surgical_probe.sh
}

run_smoke() {
  need_inputs
  mkdir -p "$OUT_ROOT/smoke"
  "$PY" tools/v13_root_mate_veto_eval.py "$EXPANDED" \
    --trajectory-json "$TRAJ" \
    --guards mate1,mate2,forcing5 \
    --top-k 6 \
    --out-json "$OUT_ROOT/smoke/mate_veto_smoke.json" \
    --out-md "$OUT_ROOT/smoke/mate_veto_smoke.md"
  "$PY" tools/v13_root_trajectory_audit.py "$EXPANDED" \
    --checkpoint "$CHECKPOINT" \
    --out-json "$OUT_ROOT/smoke/wdl_scalar_traj_smoke.json" \
    --out-md "$OUT_ROOT/smoke/wdl_scalar_traj_smoke.md" \
    --position-mode bad \
    --max-roots 2 \
    --milestones 0,32 \
    --value-sources scalar,wdl \
    --device "$DEVICE" \
    --eval-batch-size 8
}

run_mate_veto() {
  need_inputs
  mkdir -p "$OUT_ROOT/mate_veto"
  "$PY" tools/v13_root_mate_veto_eval.py "$EXPANDED" "$EXACT" \
    --trajectory-json "$TRAJ" \
    --guards mate1,mate2,forcing5,forcing7 \
    --top-k 6 \
    --out-json "$OUT_ROOT/mate_veto/mate_veto_expanded300_exact112.json" \
    --out-md "$OUT_ROOT/mate_veto/mate_veto_expanded300_exact112.md"
}

run_bad19_trajectory() {
  need_inputs
  mkdir -p "$OUT_ROOT/trajectory"
  "$PY" tools/v13_root_trajectory_audit.py "$EXPANDED" "$EXACT" \
    --checkpoint "$CHECKPOINT" \
    --out-json "$OUT_ROOT/trajectory/wdl_scalar_bad19_trajectory.json" \
    --out-md "$OUT_ROOT/trajectory/wdl_scalar_bad19_trajectory.md" \
    --position-mode bad \
    --milestones 0,64,256,1000,3000,8000 \
    --value-sources scalar,wdl \
    --device "$DEVICE" \
    --eval-batch-size 16
}

run_bad_control() {
  need_inputs
  mkdir -p "$OUT_ROOT/trajectory"
  "$PY" tools/v13_root_trajectory_audit.py "$EXPANDED" "$EXACT" \
    --checkpoint "$CHECKPOINT" \
    --out-json "$OUT_ROOT/trajectory/wdl_scalar_bad_control_final8000.json" \
    --out-md "$OUT_ROOT/trajectory/wdl_scalar_bad_control_final8000.md" \
    --position-mode bad-control \
    --controls-per-bad 2 \
    --milestones 8000 \
    --value-sources scalar,wdl \
    --device "$DEVICE" \
    --eval-batch-size 16
}

case "$PHASE" in
  static)
    run_static
    ;;
  smoke)
    run_static
    run_smoke
    ;;
  mate)
    run_static
    run_mate_veto
    ;;
  bad19)
    run_static
    run_bad19_trajectory
    ;;
  bad-control)
    run_static
    run_bad_control
    ;;
  all)
    run_static
    run_smoke
    run_mate_veto
    run_bad19_trajectory
    run_bad_control
    ;;
  *)
    printf 'unknown V135_PHASE: %s\n' "$PHASE" >&2
    printf 'valid phases: static, smoke, mate, bad19, bad-control, all\n' >&2
    exit 2
    ;;
esac
