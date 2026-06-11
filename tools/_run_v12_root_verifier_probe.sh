#!/usr/bin/env bash
# V12 root-verifier arena probes.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
cd "$REPO"

PY="${V12_VERIFIER_PY:-/home/laure/alphaxiang/venv_nospace/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

CKPT="${V12_VERIFIER_CKPT:-/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt}"
OUT_ROOT="${V12_VERIFIER_OUT_ROOT:-/home/laure/alphaxiang/v12_root_verifier_probe}"
OPENING_SUITE="${V12_VERIFIER_OPENING_SUITE:-arena_openings/human_val_opening_suite_v1.json}"
DEVICE="${V12_VERIFIER_DEVICE:-cuda:0}"
SEED="${V12_VERIFIER_SEED:-2026052401}"
MAX_OPENINGS="${V12_VERIFIER_MAX_OPENINGS:-12}"
PARALLEL_GAMES="${V12_VERIFIER_PARALLEL_GAMES:-1}"
PHASE="${V12_VERIFIER_PHASE:-paired_d5}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_file "$CKPT"
require_file "$OPENING_SUITE"

mkdir -p "$OUT_ROOT"

common_args=(
  --checkpoint "$CKPT"
  --device "$DEVICE"
  --our-side black
  --opening-suite-path "$OPENING_SUITE"
  --max-openings "$MAX_OPENINGS"
  --games-per-opening 1
  --parallel-games "$PARALLEL_GAMES"
  --opp-engine pikafish
  --opp-depth "${V12_VERIFIER_OPP_DEPTH:-5}"
  --opp-threads "${V12_VERIFIER_OPP_THREADS:-1}"
  --opp-hash-mb "${V12_VERIFIER_OPP_HASH_MB:-64}"
  --seed "$SEED"
  --our-sims "${V12_VERIFIER_SIMS:-6400}"
  --our-c-puct "${V12_VERIFIER_C_PUCT:-1.25}"
  --our-q-weight "${V12_VERIFIER_Q_WEIGHT:-1.0}"
  --our-q-clip 1.0
  --our-temperature-move "${V12_VERIFIER_TEMPERATURE:-0.1}"
)

run_case() {
  local label="$1"
  shift
  local out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"
  printf '%q ' "$PY" -u tools/external_arena.py "${common_args[@]}" "$@" --output-dir "$out_dir" > "$out_dir/command.txt"
  printf '\n' >> "$out_dir/command.txt"
  "$PY" -u tools/external_arena.py "${common_args[@]}" "$@" --output-dir "$out_dir" 2>&1 | tee "$out_dir/run.log"
}

summarize_pair() {
  "$PY" - "$OUT_ROOT/d5_baseline_${MAX_OPENINGS}" "$OUT_ROOT/d5_top6_margin120_danger100_rootmate12_${MAX_OPENINGS}" "$OUT_ROOT/d5_pair_summary_${MAX_OPENINGS}.json" "$OUT_ROOT/d5_pair_summary_${MAX_OPENINGS}.md" <<'PY'
import glob
import json
import sys
from pathlib import Path

def load_latest(root: str):
    files = sorted(glob.glob(str(Path(root) / "external_arena_*.json")))
    if not files:
        raise SystemExit(f"missing arena JSON under {root}")
    data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    return files[-1], data

def row(root: str):
    path, data = load_latest(root)
    per_game = data.get("per_game", [])
    mate_losses = sum(1 for g in per_game if g.get("result") == "opp_win" and g.get("termination") == "mate")
    return {
        "json": path,
        "games": int(data.get("games", 0)),
        "wins": int(data.get("our_wins", 0)),
        "losses": int(data.get("opp_wins", 0)),
        "draws": int(data.get("draws", 0)),
        "score_rate": float(data.get("score_rate", 0.0)),
        "mate_losses": int(mate_losses),
        "events": int((data.get("symbolic_guard_summary") or {}).get("events", 0)),
        "duration_s": float(data.get("duration_s", 0.0)),
    }

baseline = row(sys.argv[1])
gated = row(sys.argv[2])
payload = {"baseline": baseline, "gated": gated}
Path(sys.argv[3]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    "# V12 Root Verifier Paired d5 Summary",
    "",
    "| config | games | W-L-D | score | mate losses | events | duration |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, item in [("baseline", baseline), ("gated", gated)]:
    lines.append(
        f"| {name} | {item['games']} | {item['wins']}-{item['losses']}-{item['draws']} | "
        f"{100.0 * item['score_rate']:.1f}% | {item['mate_losses']} | {item['events']} | {item['duration_s']:.1f}s |"
    )
Path(sys.argv[4]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

run_baseline_d5() {
  run_case "d5_baseline_${MAX_OPENINGS}"
}

run_gated_d5() {
  run_case "d5_top6_margin120_danger100_rootmate12_${MAX_OPENINGS}" \
    --our-pikafish-verifier \
    --our-verifier-top-k 6 \
    --our-verifier-margin-cp 120 \
    --our-verifier-danger-threshold-cp 100 \
    --our-verifier-depth 8 \
    --our-verifier-threads "${V12_VERIFIER_THREADS:-4}" \
    --our-verifier-hash-mb "${V12_VERIFIER_HASH_MB:-128}" \
    --our-verifier-side black \
    --our-root-mate1-blunder-guard \
    --our-root-mate2-blunder-guard
}

case "$PHASE" in
  baseline_d5)
    run_baseline_d5
    ;;
  gated_d5)
    run_gated_d5
    ;;
  summarize)
    summarize_pair
    ;;
  paired_d5)
    run_baseline_d5
    run_gated_d5
    summarize_pair
    ;;
  *)
    echo "unknown V12_VERIFIER_PHASE=$PHASE" >&2
    echo "valid phases: baseline_d5, gated_d5, summarize, paired_d5" >&2
    exit 2
    ;;
esac
