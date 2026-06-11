#!/bin/bash
# v12.8D global-strategy snapshot smoke:
# run Pika d3/d4 first, then run d5 only if both anchors pass the promotion gate.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 CHECKPOINT OUT_DIR" >&2
    exit 2
fi

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="$1"
OUT_BASE="$2"
D3D4_GAMES="${V128_SMOKE_GAMES:-20}"
D5_GAMES="${V128_D5_GAMES:-50}"
D3_DEVICE="${V128_D3_DEVICE:-cuda:0}"
D4_DEVICE="${V128_D4_DEVICE:-cuda:1}"
D5_DEVICE="${V128_D5_DEVICE:-cuda:0}"

mkdir -p "$OUT_BASE/pika_d3" "$OUT_BASE/pika_d4" "$OUT_BASE/pika_d5"

run_external() {
    local key="$1"
    local depth="$2"
    local device="$3"
    local seed="$4"
    local games="$5"
    local out_dir="$OUT_BASE/$key"
    mkdir -p "$out_dir"
    cd "$REPO"
    "$PY" tools/external_arena.py \
        --checkpoint "$CKPT" \
        --our-sims 1600 \
        --our-c-puct 1.25 \
        --our-q-weight 1.0 \
        --our-q-clip 1.0 \
        --our-value-source scalar \
        --our-temperature-move 0.1 \
        --games "$games" \
        --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" \
        --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log" \
        | grep -E '^game [0-9]|^DONE:|score_rate|elo_estimate|symbolic_guard_summary|loaded our model|value_source'
}

run_external "pika_d3" 3 "$D3_DEVICE" 128303 "$D3D4_GAMES" &
PID3=$!
run_external "pika_d4" 4 "$D4_DEVICE" 128304 "$D3D4_GAMES" &
PID4=$!
STATUS=0
wait "$PID3" || STATUS=$?
wait "$PID4" || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "v12.8D d3/d4 smoke failed with status=$STATUS" >&2
    exit "$STATUS"
fi

"$PY" tools/summarize_panel_results.py \
    --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                    "$OUT_BASE"/pika_d4/external_arena_*.json \
    --json-out "$OUT_BASE/summary_d3d4.json" \
    --markdown-out "$OUT_BASE/summary_d3d4.md"

set +e
"$PY" - "$OUT_BASE" <<'PY'
import glob
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

def latest_score(key: str) -> tuple[float, str]:
    paths = sorted(glob.glob(str(out / key / "external_arena_*.json")))
    if not paths:
        raise SystemExit(f"missing arena json for {key}")
    path = paths[-1]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["score_rate"]), path

d3, d3_path = latest_score("pika_d3")
d4, d4_path = latest_score("pika_d4")
print(f"v12.8D gate: d3={d3:.3f} ({d3_path}) d4={d4:.3f} ({d4_path})")
if d4 < 0.30:
    print("v12.8D gate: STOP, d4 below 30%; skip d5.")
    raise SystemExit(0)
if d3 > 0.60 and d4 < 0.35:
    print("v12.8D gate: STOP, d3 spike with d4 weakness; shallow exploitation.")
    raise SystemExit(0)
if d3 >= 0.52 and d4 >= 0.39:
    print("v12.8D gate: PROMOTE, run d5 smoke.")
    raise SystemExit(2)
print("v12.8D gate: HOLD, not enough for d5.")
PY
GATE=$?
set -e

if [ "$GATE" -eq 2 ]; then
    run_external "pika_d5" 5 "$D5_DEVICE" 128305 "$D5_GAMES"
    "$PY" tools/summarize_panel_results.py \
        --external-json "$OUT_BASE"/pika_d3/external_arena_*.json \
                        "$OUT_BASE"/pika_d4/external_arena_*.json \
                        "$OUT_BASE"/pika_d5/external_arena_*.json \
        --json-out "$OUT_BASE/summary.json" \
        --markdown-out "$OUT_BASE/summary.md"
elif [ "$GATE" -eq 0 ]; then
    cp "$OUT_BASE/summary_d3d4.json" "$OUT_BASE/summary.json"
    cp "$OUT_BASE/summary_d3d4.md" "$OUT_BASE/summary.md"
else
    echo "v12.8D gate failed with status=$GATE" >&2
    exit "$GATE"
fi

echo "v12.8D global-strategy smoke DONE: $OUT_BASE"
