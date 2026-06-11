#!/bin/bash
# Lightweight v13 health monitor.  It only observes and logs; it never kills or
# modifies training processes.

set -euo pipefail

LOG_ROOT="${V13_LOG_ROOT:-/home/laure/alphaxiang/v13_logs}"
TRAIN_PID_FILE="$LOG_ROOT/v13_200m_serial_train.pid"
TRAIN_LOG="$LOG_ROOT/v13_200m_serial_train.log"
HEALTH_LOG="$LOG_ROOT/v13_health_monitor.log"
INTERVAL="${V13_MONITOR_INTERVAL_SECONDS:-300}"

mkdir -p "$LOG_ROOT"

latest_train_line() {
    if [ -f "$TRAIN_LOG" ]; then
        grep 'train step=' "$TRAIN_LOG" | tail -1 || true
    fi
}

latest_eval_line() {
    if [ -f "$TRAIN_LOG" ]; then
        grep 'human_val_total_loss=' "$TRAIN_LOG" | tail -1 || true
    fi
}

pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

while true; do
    {
        echo "==== $(date -Is) ===="
        if [ -f "$TRAIN_PID_FILE" ]; then
            TRAIN_PID="$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)"
        else
            TRAIN_PID=""
        fi
        if pid_alive "$TRAIN_PID"; then
            echo "train_root_pid=$TRAIN_PID status=alive"
            ps --forest -o pid,ppid,stat,etime,cmd -g "$TRAIN_PID" | head -20 || true
        else
            echo "train_root_pid=${TRAIN_PID:-missing} status=not_alive"
        fi
        echo "latest_train=$(latest_train_line)"
        echo "latest_eval=$(latest_eval_line)"
        echo "gpu:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits || true
        echo "watchers:"
        for name in dense strategy; do
            PID_FILE="$LOG_ROOT/v13_${name}_smoke_watcher.pid"
            PID="$(cat "$PID_FILE" 2>/dev/null || true)"
            if pid_alive "$PID"; then
                echo "${name}_watcher_pid=$PID status=alive"
            else
                echo "${name}_watcher_pid=${PID:-missing} status=not_alive"
            fi
        done
        echo "snapshots:"
        find /home/laure/alphaxiang/training_runs/run_020a_v13_200m_dense_baseline/snapshots \
             /home/laure/alphaxiang/training_runs/run_020b_v13_200m_strategy_tokens/snapshots \
             -maxdepth 1 -type f -name 'latest_step*.pt' 2>/dev/null | sort | tail -10 || true
        echo
    } >> "$HEALTH_LOG"
    sleep "$INTERVAL"
done
