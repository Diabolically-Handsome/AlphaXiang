#!/usr/bin/env bash
# Watchdog for the decoupled overnight setup: trainer (run_056) + builder (run_057).
# Auto-restart either if it dies. Capped to avoid infinite crash loops.
set -uo pipefail
TRUN="/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer"
BRUN="/home/laure/alphaxiang/training_runs/run_057_v2redo_builder"
WLOG="/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer/watchdog2.log"
TRAINER_SH="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_trainer.sh"
BUILDER_SH="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_builder.sh"

t_restarts=0; b_restarts=0; MAX=6
log(){ echo "[wd2 $(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }
log "watchdog2 started"
sleep 120   # let both come up first
while true; do
  t_alive=$(pgrep -fc 'run_056_v2redo_trainer' || true)
  b_alive=$(pgrep -fc 'run_057_v2redo_builder' || true)
  # trainer down? (only restart if it hasn't simply finished — check max-steps not reached)
  if [[ "${t_alive:-0}" -eq 0 ]]; then
    if grep -q 'training completed normally' "$TRUN/trainer.log" 2>/dev/null; then
      : # finished cleanly, do not restart
    elif [[ "$t_restarts" -lt "$MAX" ]]; then
      t_restarts=$((t_restarts+1)); log "trainer down -> restart #$t_restarts"
      nohup bash "$TRAINER_SH" >> "$TRUN/restart.log" 2>&1 &
      sleep 60
    else
      log "trainer gave up (>$MAX restarts)"
    fi
  fi
  if [[ "${b_alive:-0}" -eq 0 ]]; then
    if [[ "$b_restarts" -lt "$MAX" ]]; then
      b_restarts=$((b_restarts+1)); log "builder down -> restart #$b_restarts"
      nohup bash "$BUILDER_SH" >> "$BRUN/restart.log" 2>&1 &
      sleep 60
    else
      log "builder gave up (>$MAX restarts)"
    fi
  fi
  sleep 120
done
