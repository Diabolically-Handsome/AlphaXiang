#!/usr/bin/env bash
# Overnight watchdog for the d20 Stage-2 redo (run_051).
# Every CHECK_INTERVAL: verify the driver is alive AND driver.log is advancing.
# If down/stalled, resume (no buffer reset). Cap restarts to avoid an infinite crash loop.
set -uo pipefail

RUNDIR="/home/laure/alphaxiang/training_runs/run_051_v2redo_d20_from_stage1"
DRIVERLOG="$RUNDIR/driver.log"
WLOG="$RUNDIR/watchdog.log"
RESUME="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/_run_v2redo_d20_resume.sh"
MATCH="stage1_driver.py.*run_051_v2redo_d20_from_stage1"

CHECK_INTERVAL=120          # seconds between checks
STALL_SECS=2400             # 40 min with no driver.log write = stalled
MAX_RESTARTS=6              # give up after this many restarts
restarts=0

log(){ echo "[watchdog $(date +%H:%M:%S)] $*" | tee -a "$WLOG"; }

log "watchdog started (interval=${CHECK_INTERVAL}s stall=${STALL_SECS}s max_restarts=${MAX_RESTARTS})"

while true; do
  alive=$(pgrep -fc "$MATCH" || true)
  # log mtime age
  if [[ -f "$DRIVERLOG" ]]; then
    now=$(date +%s); mt=$(stat -c %Y "$DRIVERLOG" 2>/dev/null || echo "$now")
    age=$(( now - mt ))
  else
    age=0
  fi

  need_restart=0; reason=""
  if [[ "${alive:-0}" -eq 0 ]]; then need_restart=1; reason="driver_not_running"; fi
  if [[ "${alive:-0}" -ge 1 && "$age" -gt "$STALL_SECS" ]]; then need_restart=1; reason="stalled_${age}s"; fi

  if [[ "$need_restart" -eq 1 ]]; then
    if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
      log "GIVING UP after $restarts restarts (last reason=$reason). Leaving down for manual diagnosis."
      exit 1
    fi
    # If stalled but process alive, kill the stuck driver subtree first
    if [[ "${alive:-0}" -ge 1 ]]; then
      log "stalled (age=${age}s); killing stuck driver before resume"
      for p in $(pgrep -f "$MATCH"); do kill -9 "$p" 2>/dev/null || true; done
      pkill -9 -f 'xiangqi_train.py' 2>/dev/null || true
      pkill -9 -f 'pikafish' 2>/dev/null || true
      sleep 5
    fi
    restarts=$(( restarts + 1 ))
    log "RESTART #$restarts (reason=$reason) -> resume (no reset)"
    nohup bash "$RESUME" >> "$RUNDIR/resume_launch.log" 2>&1 &
    sleep 90   # give it time to come up before next check
  fi

  sleep "$CHECK_INTERVAL"
done
