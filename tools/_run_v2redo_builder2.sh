#!/usr/bin/env bash
# CPU-only d20 data builder: generate (Pikafish d6 self-play, no GPU) -> label value d20 ->
# label policy d20 -> PUBLISH the fully-labeled dir into the big pool. No training => no OOM,
# no train-fail-skips-label. Loops forever, growing the pool.
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

PUBLISHED="/home/laure/alphaxiang/selfplay_runs_v2redo_d20_big"
STAGEROOT="/home/laure/alphaxiang/v2redo_staging"
LOG="$PUBLISHED/builder.log"
mkdir -p "$PUBLISHED" "$STAGEROOT"

NPOS="${V2REDO_BUILD_NPOS:-3000}"
# Per-launch seed base so a restart does NOT re-generate the same positions (was: 20000+i from i=1,
# which duplicated the first ~15 batches on every restart). date +%s makes each launch fresh.
SEEDBASE="${V2REDO_SEEDBASE:-$(date +%s)}"
log(){ echo "[builder $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "builder2 started (npos=$NPOS per batch, seedbase=$SEEDBASE)"

i=0
while true; do
  i=$((i+1))
  ts=$(date +%Y%m%d_%H%M%S)
  STAGE="$STAGEROOT/batch_${ts}_$i"
  mkdir -p "$STAGE/train"

  log "batch $i: generate $NPOS distill positions (Pikafish d6, CPU)"
  if ! "$PY" tools/distillation_generator.py \
        --output-dir "$STAGE/train" --num-positions "$NPOS" --depth 6 \
        --workers 16 --threads-per-worker 1 --hash-mb 16 --shard-size 2048 \
        --seed $((SEEDBASE + i)) --random-opening-plies 20 >> "$LOG" 2>&1; then
    log "batch $i: generate FAILED; skipping"; rm -rf "$STAGE"; sleep 10; continue
  fi

  log "batch $i: label VALUE d20 (28 workers)"
  if ! "$PY" tools/oracle_value_labeler.py \
        --input-shard-dir "$STAGE/train" --depth 20 --workers 28 --hash-mb 64 \
        --max-wait-per-shard-s 3600 >> "$LOG" 2>&1; then
    log "batch $i: value label FAILED; skipping"; rm -rf "$STAGE"; sleep 10; continue
  fi

  log "batch $i: label POLICY d20 multipv=6 (28 workers, 7970X)"
  if ! "$PY" tools/oracle_policy_labeler.py \
        --input-shard-dir "$STAGE/train" --depth 20 --multipv 6 \
        --adaptive-temperature --legal-smoothing 0.05 --workers 28 --hash-mb 64 \
        --max-wait-per-shard-s 3600 >> "$LOG" 2>&1; then
    log "batch $i: policy label FAILED; skipping"; rm -rf "$STAGE"; sleep 10; continue
  fi

  # publish: move fully-labeled dir into the big pool (atomic-ish)
  dest="$PUBLISHED/batch_${ts}_$i"
  mv "$STAGE" "$dest"
  nsh=$(ls "$dest/train"/shard_*.pt 2>/dev/null | wc -l)
  log "batch $i: PUBLISHED -> $dest ($nsh shards)"
done
