#!/usr/bin/env bash
# Watchdog for the scaled d20 redo: builder (run_v2redo_builder2) + supervisor (run_v2redo_supervisor).
# Auto-restart either if its top-level loop process dies. Capped to avoid crash loops.
set -uo pipefail
WLOG="/home/laure/alphaxiang/training_runs/run_058_v2redo_bigtrain/watchdog.log"
BUILDER="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_builder2.sh"
SUPER="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_supervisor.sh"
b_re=0; s_re=0; MAX=10
log(){ echo "[wd $(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }
log "big-redo watchdog started"
sleep 150
while true; do
  b=$(pgrep -fc '_run_v2redo_builder2.sh' || true)
  s=$(pgrep -fc '_run_v2redo_supervisor.sh' || true)
  if [[ "${b:-0}" -eq 0 && "$b_re" -lt "$MAX" ]]; then
    b_re=$((b_re+1)); log "builder down -> restart #$b_re"
    nohup bash "$BUILDER" >/dev/null 2>&1 &
    sleep 30
  fi
  if [[ "${s:-0}" -eq 0 && "$s_re" -lt "$MAX" ]]; then
    s_re=$((s_re+1)); log "supervisor down -> restart #$s_re"
    nohup bash "$SUPER" >/dev/null 2>&1 &
    sleep 30
  fi
  sleep 120
done
