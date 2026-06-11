#!/usr/bin/env bash
# Probe: did d20 training from Stage-1 move strength? Compare 3 checkpoints at identical protocol.
# Weak-model-friendly opponents (Pika d1+noise and d2) since Stage-1 is ~v3-level.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

OUT=/home/laure/alphaxiang/v2redo_probe
STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
# final trained model = run_056 latest.pt (or the highest snapshot)
TRAINED="/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer/latest.pt"
mkdir -p "$OUT"

run() {
  local label="$1"; local ckpt="$2"; local depth="$3"; local noise="$4"; local dev="$5"; local seed="$6"
  local d="$OUT/${label}_d${depth}n${noise}"
  mkdir -p "$d"
  local extra=""
  [[ "$noise" != "0" ]] && extra="--opp-noise-ratio 0.${noise}"
  "$PY" -u tools/external_arena.py \
    --checkpoint "$ckpt" --device "$dev" \
    --games 30 --parallel-games 4 --our-sims 1600 \
    --our-c-puct 1.25 --our-q-weight 1.0 --our-temperature-move 0.1 \
    --opp-engine pikafish --opp-depth "$depth" $extra \
    --seed "$seed" --output-dir "$d" \
    2>&1 | tee "$d/run.log" | grep -E '^DONE:|score_rate|loaded our model' || true
}

echo "[probe] start $(date +%H:%M:%S)"
# Stage-1 baseline vs d1+noise and d2 (cuda:1)
run "stage1"  "$STAGE1"  1 15 cuda:1 61001
run "stage1"  "$STAGE1"  2 0  cuda:1 61002
# d20-trained vs d1+noise and d2 (cuda:0)
run "trained" "$TRAINED" 1 15 cuda:0 61001
run "trained" "$TRAINED" 2 0  cuda:0 61002
echo "[probe] done $(date +%H:%M:%S)"
echo "=== SUMMARY ==="
for f in "$OUT"/*/run.log; do echo "$f: $(grep score_rate "$f" | tail -1)"; done
