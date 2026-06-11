#!/bin/bash
# After Light step300000 d5 finishes, stop the slower full smoke queue and
# prioritize d5-only checks for later Light snapshots.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
OUT_ROOT="/home/laure/alphaxiang/v127_snapshot_smoke"
STEP300="$OUT_ROOT/light_step300000"
D5_QUEUE_LOG="$OUT_ROOT/light_d5_queue.log"

while ! compgen -G "$STEP300/pika_d5/external_arena_*.json" >/dev/null; do
    sleep 60
done

echo "step300000 d5 JSON detected; stopping full Light snapshot queue"
pkill -f "_run_v127_light_snapshot_queue.sh" || true
pkill -f "$STEP300/pika_d4" || true
pkill -f "$STEP300/pika_d3" || true

"$PY" "$REPO/tools/summarize_panel_results.py" \
    --external-json "$STEP300"/pika_d5/external_arena_*.json \
    --json-out "$STEP300/summary_d5_only.json" \
    --markdown-out "$STEP300/summary_d5_only.md"

cd "$REPO"
nohup bash tools/_run_v127_light_d5_queue.sh >> "$D5_QUEUE_LOG" 2>&1 < /dev/null &
echo "started d5-only Light snapshot queue; log=$D5_QUEUE_LOG"
