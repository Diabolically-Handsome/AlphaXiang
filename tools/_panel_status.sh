#!/bin/bash
echo "=== panel arena progress ==="
date
for d in /home/laure/alphaxiang/arena_runs/full_panel/v*/; do
    name=$(basename "$d")
    last=$(tail -3 "$d/run.log" 2>/dev/null | grep -E 'game [0-9]|DONE' | tail -1)
    if [ -z "$last" ]; then
        last="(no progress)"
    fi
    echo "$name: $last"
done
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
