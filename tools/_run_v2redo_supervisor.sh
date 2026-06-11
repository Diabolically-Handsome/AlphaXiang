#!/usr/bin/env bash
# Trainer supervisor for the scaled d20 redo. Each block: re-ingest the FULL current pool with
# --replay-buffer-size == pool size (capped) so buffer_fill=1.0 => oracle_cov ~0.95, train a block
# of steps, then loop (the builder has grown the pool => bigger buffer next block). Resumes from
# latest.pt each block (model progress accumulates).
set -uo pipefail
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/alphaxiang/venv_nospace/bin/python"
[[ -x "$PY" ]] || PY="python3"
cd "$REPO"

POOL="/home/laure/alphaxiang/selfplay_runs_v2redo_d20_big"
STAGE1="/home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt"
RUNDIR="/home/laure/alphaxiang/training_runs/run_058_v2redo_bigtrain"
LOG="$RUNDIR/supervisor.log"
BLOCK="${V2REDO_BLOCK_STEPS:-4000}"
BUFCAP="${V2REDO_BUFCAP:-60000}"

mkdir -p "$RUNDIR"
[[ -f "$RUNDIR/latest.pt" ]] || { echo "seeding from Stage-1 181000"; cp "$STAGE1" "$RUNDIR/latest.pt"; }
log(){ echo "[sup $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "supervisor started (block=$BLOCK bufcap=$BUFCAP)"

pool_samples() {
  "$PY" - "$POOL" <<'PYEOF'
import sys, json, glob, os
tot=0
for m in glob.glob(os.path.join(sys.argv[1], "*", "manifest.json")):
    try:
        d=json.load(open(m))
        tot += int(d.get("total_samples_written") or d.get("total_samples") or 0)
    except Exception: pass
print(tot)
PYEOF
}
cur_step() { "$PY" - "$RUNDIR/latest.pt" <<'PYEOF'
import sys, torch
try: print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("global_step",181000)))
except Exception: print(181000)
PYEOF
}

while true; do
  P=$(pool_samples)
  [[ -z "$P" || "$P" -lt 2000 ]] && { log "pool too small ($P); wait 120s"; sleep 120; continue; }
  buf=$P; [[ "$buf" -gt "$BUFCAP" ]] && buf=$BUFCAP
  step=$(cur_step); target=$((step + BLOCK))
  log "block: pool=$P buffer=$buf step=$step -> $target"
  "$PY" -u xiangqi_train.py \
    --foreground \
    --human-data-dir /home/laure/alphaxiang/human_bootstrap_data_elite_wdl \
    --selfplay-dirs "$POOL" \
    --output-dir "$RUNDIR" \
    --resume-path "$RUNDIR/latest.pt" \
    --device cuda:0 \
    --max-steps "$target" \
    --lr-schedule-max-steps 400000 \
    --learning-rate 2e-4 \
    --replay-buffer-size "$buf" \
    --log-interval-steps 100 \
    --eval-interval-steps 1000 \
    --save-interval-steps 1000 \
    --snapshot-interval-steps 2000 \
    --disable-selfplay-run-quality-gate \
    --reset-selfplay-ingest-state-on-resume \
    --bootstrap-human-floor 0.05 \
    --wdl-loss-weight 1.0 --value-loss-weight 0.5 --value-target-scale 0.9 \
    --use-oracle-value --policy-oracle-alpha 0.5 \
    >> "$RUNDIR/train_block.log" 2>&1 || { log "train block exited nonzero; wait 30s"; sleep 30; }
  log "block done at step ~$target"
done
