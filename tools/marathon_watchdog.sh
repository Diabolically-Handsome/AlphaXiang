#!/bin/bash
# Watchdog for marathon training: detects log stalls (>STALL_MIN minutes without log writes)
# and forcibly restarts the training process.  Lives inside WSL.
#
# Usage: marathon_watchdog.sh <log_path> <stall_minutes>
# Assumes the training command is baked into RELAUNCH_CMD below.

set -u

LOG_FILE="${1:?missing log path}"
STALL_MIN="${2:-15}"
WATCHDOG_LOG="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/marathon_watchdog.log"
REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
DATA_ROOT="/home/laure/alphaxiang"
VENV_ACTIVATE="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/activate"

RELAUNCH_CMD="python -u xiangqi_closed_loop.py \
  --cycles 0 \
  --execution-mode overlap \
  --human-data-dir $DATA_ROOT/human_bootstrap_data_elite_wdl \
  --training-output-dir $DATA_ROOT/training_runs/run_001 \
  --selfplay-output-root $DATA_ROOT/selfplay_runs \
  --arena-output-root $DATA_ROOT/arena_runs \
  --selfplay-target-samples-per-cycle 12000 \
  --train-steps-per-cycle 2000 \
  --train-lr-schedule-max-steps 200000 \
  --overlap-selfplay-num-workers 14 \
  --selfplay-eval-batch-size 64 \
  --selfplay-max-states-per-batch 384 \
  --selfplay-checkpoint-source latest \
  --train-cpu-sampler-workers 24 \
  --train-cpu-prefetch-batches 20 \
  --overlap-keep-free-cpu-cores 2 \
  --arena-games 100 \
  --arena-sims 800 \
  --arena-games-per-opening 2 \
  --arena-min-non-draw-games 3 \
  --arena-temperature-move 0.5 \
  --device cuda:0 \
  --pause-at-local-time off"

exec >> "$WATCHDOG_LOG" 2>&1
echo ""
echo "=== watchdog started $(date) pid=$$ log=$LOG_FILE stall_min=$STALL_MIN ==="

restart_count=0

find_pid() {
    pgrep -f 'xiangqi_closed_loop.py' | head -1
}

while true; do
    sleep 60
    pid=$(find_pid)
    if [ -z "$pid" ]; then
        echo "[$(date '+%H:%M:%S')] no xiangqi_closed_loop running; relaunching (restart #$((++restart_count)))"
        # Restore latest.pt from best.pt if corrupt/missing
        cd "$DATA_ROOT/training_runs/run_001" || continue
        if [ ! -f latest.pt ] || ! python -c "import torch; torch.load('latest.pt', map_location='cpu', weights_only=False)" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] latest.pt missing or corrupt; restoring from best.pt"
            [ -f latest.pt ] && mv latest.pt "latest_corrupt_watchdog_$(date +%Y%m%d_%H%M%S).pt"
            cp -p best.pt latest.pt
        fi
        cd "$REPO"
        source "$VENV_ACTIVATE"
        nohup $RELAUNCH_CMD > "$LOG_FILE" 2>&1 < /dev/null &
        disown
        sleep 30
        continue
    fi

    # check log mtime
    if [ ! -f "$LOG_FILE" ]; then
        echo "[$(date '+%H:%M:%S')] log $LOG_FILE missing; ignoring"
        continue
    fi
    mtime=$(stat -c '%Y' "$LOG_FILE")
    now=$(date +%s)
    age_sec=$((now - mtime))
    age_min=$((age_sec / 60))

    if [ "$age_min" -ge "$STALL_MIN" ]; then
        echo "[$(date '+%H:%M:%S')] STALL DETECTED: log age ${age_min}min >= ${STALL_MIN}min; killing pid=$pid and children (restart #$((++restart_count)))"
        # send SIGKILL to all xiangqi-related processes
        pgrep -f 'xiangqi_closed_loop|xiangqi_train|xiangqi_selfplay|xiangqi_arena' | xargs -r kill -9 2>/dev/null
        pgrep -f 'multiprocessing.spawn|multiprocessing.resource_tracker' | xargs -r kill -9 2>/dev/null
        sleep 10
        # forced recovery
        cd "$DATA_ROOT/training_runs/run_001" || continue
        if [ ! -f latest.pt ] || ! python -c "import torch; torch.load('latest.pt', map_location='cpu', weights_only=False)" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] latest.pt corrupt after kill; restoring from best.pt"
            [ -f latest.pt ] && mv latest.pt "latest_corrupt_watchdog_$(date +%Y%m%d_%H%M%S).pt"
            cp -p best.pt latest.pt
        fi
        cd "$REPO"
        source "$VENV_ACTIVATE"
        nohup $RELAUNCH_CMD > "$LOG_FILE" 2>&1 < /dev/null &
        disown
        sleep 30
    fi
done
