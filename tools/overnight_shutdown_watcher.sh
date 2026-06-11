#!/bin/bash
# Watches the overnight training process and shuts down the Windows host after it exits.
# Safe-guards: only triggers shutdown if the process exits AFTER the earliest-allowed hour.

set -u

TRAIN_PID="${1:?missing training PID}"
LOG_FILE="${2:?missing log file path}"
EARLIEST_SHUTDOWN_HOUR="${3:-6}"  # earliest local hour at which shutdown is allowed

WATCHER_LOG="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/tools/overnight_watcher.log"

exec >> "$WATCHER_LOG" 2>&1

echo "=== watcher started at $(date) pid=$$ watching train_pid=$TRAIN_PID earliest_hour=$EARLIEST_SHUTDOWN_HOUR ==="

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

EXIT_TIME="$(date '+%Y-%m-%d %H:%M:%S %Z')"
EXIT_HOUR="$(date '+%H')"

echo "=== training process $TRAIN_PID exited at $EXIT_TIME ==="
echo "=== final log tail ==="
tail -30 "$LOG_FILE"
echo "=== end of log tail ==="

echo "--- checkpoint state ---"
ls -la "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/training_runs/run_001/latest.pt" \
       "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer/training_runs/run_001/best.pt" 2>&1

if [ "$EXIT_HOUR" -lt "$EARLIEST_SHUTDOWN_HOUR" ]; then
    echo "WARNING: process exited at hour $EXIT_HOUR which is before earliest allowed hour $EARLIEST_SHUTDOWN_HOUR; SKIPPING shutdown"
    echo "(leaving machine on for investigation)"
    exit 1
fi

# Wait 3 min to let any pending disk flush / OS buffers settle before shutting down.
echo "=== waiting 180s before invoking Windows shutdown ==="
sleep 180

echo "=== invoking Windows shutdown at $(date) ==="
cmd.exe /c "shutdown /s /t 60 /c \"AlphaXiang overnight training finished; shutting down in 60 seconds\"" 2>&1
echo "=== shutdown command issued; exiting watcher ==="
