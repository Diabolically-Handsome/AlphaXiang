#!/bin/bash
# 10-minute GPU + process logger for diagnosing dual-GPU stutter while gaming.
# Samples nvidia-smi every 1s (gpu state) and every 5s (compute processes).

OUT_DIR="${1:-/tmp/game_stutter}"
DURATION="${2:-600}"

mkdir -p "$OUT_DIR"
GPU_LOG="$OUT_DIR/gpu_state.csv"
PROC_LOG="$OUT_DIR/processes.csv"
DISPLAY_LOG="$OUT_DIR/display.txt"

# Header
echo "ts_ms,gpu_idx,util_gpu,util_mem,mem_used_mib,power_w,temp_c,fan_pct,clk_gfx_mhz,clk_mem_mhz" > "$GPU_LOG"
echo "ts_ms,pid,process_name,gpu_idx,used_mem_mib" > "$PROC_LOG"

# One-time: which GPU has display attached?
{
  echo "=== display state at start (epoch: $(date +%s)) ==="
  nvidia-smi --query-gpu=index,name,display_mode,display_active,vbios_version --format=csv
  echo ""
  echo "=== driver / CUDA versions ==="
  nvidia-smi --query-gpu=index,driver_version --format=csv
} > "$DISPLAY_LOG"

START=$SECONDS
END=$((SECONDS + DURATION))
TICK=0

while [ $SECONDS -lt $END ]; do
    TS=$(date +%s%3N)

    # Per-GPU sample (1s cadence)
    nvidia-smi \
      --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,fan.speed,clocks.current.graphics,clocks.current.memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$TS" '{print ts","$0}' >> "$GPU_LOG"

    # Process list (5s cadence; processes don't churn fast)
    if [ $((TICK % 5)) -eq 0 ]; then
        nvidia-smi \
          --query-compute-apps=pid,process_name,gpu_bus_id,used_memory \
          --format=csv,noheader,nounits 2>/dev/null \
          | awk -v ts="$TS" '{print ts","$0}' >> "$PROC_LOG"
    fi

    TICK=$((TICK + 1))
    sleep 1
done

echo "DONE. Logged $(wc -l < "$GPU_LOG") GPU samples and $(wc -l < "$PROC_LOG") process samples."
echo "Files:"
echo "  $GPU_LOG"
echo "  $PROC_LOG"
echo "  $DISPLAY_LOG"
