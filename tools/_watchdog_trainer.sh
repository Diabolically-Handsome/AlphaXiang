#!/usr/bin/env bash
# Watchdog for the overnight TRAINER (run_056). Restart if it dies before completing max-steps.
set -uo pipefail
TRUN="/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer"
WLOG="$TRUN/watchdog.log"
TRAINER_SH="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_trainer.sh"
restarts=0; MAX=6
log(){ echo "[wd $(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }
log "trainer watchdog started"
sleep 90
while true; do
  alive=$(pgrep -fc 'run_056_v2redo_trainer' || true)
  if [[ "${alive:-0}" -eq 0 ]]; then
    if grep -q 'training completed normally' "$TRUN/trainer.log" 2>/dev/null; then
      log "trainer finished normally; watchdog exiting"; exit 0
    elif [[ "$restarts" -lt "$MAX" ]]; then
      restarts=$((restarts+1)); log "trainer down (not finished) -> restart #$restarts"
      nohup bash "$TRAINER_SH" >> "$TRUN/restart.log" 2>&1 &
      sleep 90
    else
      log "gave up after $MAX restarts"; exit 1
    fi
  fi
  sleep 120
done
