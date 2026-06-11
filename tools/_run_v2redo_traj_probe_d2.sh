#!/usr/bin/env bash
# Lightweight trajectory probe (GPU1, builder keeps running): only d1+noise and d2 (these
# complete even under CPU contention from the 24-worker builder; d3 stalls so it's deferred).
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"
RUN=/home/laure/alphaxiang/training_runs/run_058_v2redo_bigtrain
CKPT=$(ls -t "$RUN"/snapshots/latest_step*.pt 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && CKPT="$RUN/latest.pt"
STEP=$(basename "$CKPT" | grep -oE 'step[0-9]+' || echo latest)
OUT=/home/laure/alphaxiang/v2redo_traj/$STEP
mkdir -p "$OUT"
echo "[traj-d2] probing $CKPT ($STEP) at $(date +%H:%M)"
run() {
  local depth="$1"; local noise="$2"; local seed="$3"
  local d="$OUT/d${depth}n${noise}"; mkdir -p "$d"
  local extra=""; [[ "$noise" != "0" ]] && extra="--opp-noise-ratio 0.${noise}"
  "$PY" -u tools/external_arena.py --checkpoint "$CKPT" --device cuda:1 \
    --games 30 --parallel-games 4 --our-sims 1600 --our-c-puct 1.25 --our-q-weight 1.0 \
    --our-temperature-move 0.1 --opp-engine pikafish --opp-depth "$depth" $extra \
    --seed "$seed" --output-dir "$d" 2>&1 | tee "$d/run.log" | grep -E '^DONE:|score_rate' || true
}
run 1 15 72001
run 2 0  72002
echo "=== TRAJ-d2 $STEP ($(date +%H:%M)) ==="
for dd in d1n15 d2n0; do echo "$dd: $(grep score_rate "$OUT/$dd/run.log" 2>/dev/null | tail -1)"; done
