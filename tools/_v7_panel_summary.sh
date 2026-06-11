#!/bin/bash
echo "=== v7 vs Pikafish d=1+n0.15 ==="
tail -10 /home/laure/alphaxiang/arena_runs/v7_panel/v7_pika_d1n15/run.log 2>/dev/null | grep -E 'DONE:|score_rate|elo_estimate'
echo
echo "=== v7 vs Pikafish d=3 ==="
tail -10 /home/laure/alphaxiang/arena_runs/v7_panel/v7_pika_d3/run.log 2>/dev/null | grep -E 'DONE:|score_rate|elo_estimate'
echo
echo "=== v7 vs Fairy-SF d=3 ==="
tail -10 /home/laure/alphaxiang/arena_runs/v7_panel/v7_fairy_d3/run.log 2>/dev/null | grep -E 'DONE:|score_rate|elo_estimate'
echo
echo "=== v7 vs CNN ==="
tail -15 /home/laure/alphaxiang/arena_runs/v7_panel/v7_cnn/run.log 2>/dev/null | grep -E 'FINAL:|score_rate|decisive'
