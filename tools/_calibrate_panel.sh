#!/bin/bash
# Calibrate held-out evaluation panel by running v6 against each engine at
# multiple difficulty levels. Goal: find a setting per engine where v6 score
# rate lands in [30%, 70%] — that's where the engine is sensitive enough to
# detect future improvements (or regressions) without being unbeatable or
# trivial.
#
# All tests share: checkpoint, our_sims=800, parallel-games where safe,
# 10 games each, deterministic seed (different per test for variety).

set -e

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/PEAK_step210000_v6_probe2_score65pct_d3.pt"
OUT="/tmp/calibrate_panel"
GAMES=10
SIMS=800

mkdir -p "$OUT"
echo "checkpoint: $CKPT  games=$GAMES  our_sims=$SIMS"
echo ""

run_test() {
    local name="$1"; shift
    echo "============================================================"
    echo "RUNNING: $name"
    echo "============================================================"
    cd "$REPO"
    "$PY" tools/external_arena.py "$@" \
        --checkpoint "$CKPT" \
        --our-sims "$SIMS" \
        --games "$GAMES" \
        --output-dir "$OUT/$name" \
        2>&1 | tail -8
    echo ""
}

# Pikafish at different depths.  We have d=1+n0.15 (~65%) and d=8 (0%) baselines.
# Goal: find where v6 sits in 30-70% range.
run_test "pikafish_d3" \
    --opp-engine pikafish --opp-depth 3 \
    --parallel-games 4 --device cuda:0 --seed 1001

run_test "pikafish_d5" \
    --opp-engine pikafish --opp-depth 5 \
    --parallel-games 4 --device cuda:0 --seed 1002

# Fairy-Stockfish: NNUE+alphabeta but different network, totally uncalibrated.
run_test "fairy_sf_d2" \
    --opp-engine fairy_sf --opp-depth 2 \
    --parallel-games 4 --device cuda:0 --seed 2001

run_test "fairy_sf_d3" \
    --opp-engine fairy_sf --opp-depth 3 \
    --parallel-games 4 --device cuda:0 --seed 2002

# ElephantArt: AlphaZero CNN+MCTS, totally uncalibrated.  Slow per-game, so
# parallel-games=1 to keep memory manageable.  --opp-depth is ignored for ElephantArt
# (it's bounded by --elephantart-playouts) but external_arena needs *some* depth knob set.
run_test "elephantart_p400" \
    --opp-engine elephantart --opp-depth 1 \
    --elephantart-playouts 400 \
    --parallel-games 1 --device cuda:0 --seed 3001

# ============================================================
# Summary parser — read each tournament JSON, print a clean table.
# ============================================================
echo "============================================================"
echo "CALIBRATION SUMMARY — v6 (step 210K) vs each panel candidate"
echo "============================================================"

"$PY" - <<'PYEOF'
import json
import os
from pathlib import Path

OUT = Path("/tmp/calibrate_panel")
print(f"{'engine setting':<22} {'W-L-D':>10} {'score%':>8} {'Elo':>7}  {'avg_plies':>9}  {'duration':>9}")
print("-" * 78)

tests = [
    ("pikafish_d3",      "Pikafish d=3"),
    ("pikafish_d5",      "Pikafish d=5"),
    ("fairy_sf_d2",      "Fairy-SF d=2"),
    ("fairy_sf_d3",      "Fairy-SF d=3"),
    ("elephantart_p400", "ElephantArt p=400"),
]
for slug, label in tests:
    d = OUT / slug
    if not d.exists():
        print(f"{label:<22}  (no output)")
        continue
    files = sorted(d.glob("external_arena_*.json"))
    if not files:
        print(f"{label:<22}  (no JSON)")
        continue
    j = json.loads(files[-1].read_text())
    w, l, dr = j["our_wins"], j["opp_wins"], j["draws"]
    sr = j["score_rate"] * 100
    elo = j.get("elo_estimate", 0.0)
    plies = j.get("avg_plies", 0)
    dt = j.get("duration_s", 0)
    flag = "  ← in band" if 30 <= sr <= 70 else ""
    print(f"{label:<22} {w:2d}-{l:2d}-{dr:2d}    {sr:6.1f}%  {elo:+5.0f}    {plies:5.1f}    {dt:4.0f}s{flag}")

print()
print("'in band' = v6 score rate in [30%, 70%] = sensitive panel slot")
PYEOF
