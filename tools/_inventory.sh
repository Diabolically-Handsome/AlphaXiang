#!/bin/bash
# Inventory all .pt / .pth model files (skip shard data)
echo "=== checkpoint inventory ==="
find /home/laure/alphaxiang/ \( -name "*.pt" -o -name "*.pth" \) 2>/dev/null \
  | grep -v shard_ | grep -v selfplay_runs \
  | while read p; do
      size=$(du -h "$p" | cut -f1)
      echo "$size  $p"
    done

echo ""
echo "=== total disk used by training_runs ==="
du -sh /home/laure/alphaxiang/training_runs/* 2>/dev/null | sort -h
