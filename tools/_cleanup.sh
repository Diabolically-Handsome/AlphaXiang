#!/bin/bash
# Surgical cleanup: keep only step 196K + Stage 1 best.pt as references.
# Delete all other historical model files and old selfplay shards.

set -e

KEPT=()
FREED_KB=0

del_with_log() {
  if [ -e "$1" ]; then
    sz=$(du -sk "$1" | cut -f1)
    rm -rf "$1"
    FREED_KB=$((FREED_KB + sz))
    echo "  freed: $1  ($((sz / 1024)) MB)"
  fi
}

echo "=== Phase 1: prune run_006_stage2_v4/snapshots/ (keep only step 196K) ==="
for f in /home/laure/alphaxiang/training_runs/run_006_stage2_v4/snapshots/latest_step*.pt; do
  if [[ "$f" == *step196000.pt ]]; then
    KEPT+=("$f")
    echo "  kept: $f (PEAK)"
  else
    del_with_log "$f"
  fi
done

echo ""
echo "=== Phase 2: delete forensic snapshots from run_006_v4 ==="
del_with_log /home/laure/alphaxiang/training_runs/run_006_stage2_v4/snapshot_step211000_halted_after_regression.pt

echo ""
echo "=== Phase 3: delete entire failed/old training run dirs ==="
for d in run_001 run_003_stage2_FAILED_pessimism_collapse run_004_stage2_v2 \
         run_005_stage2_v22_ladder run_005_stage2_v3 \
         run_006_stage2_v23_controls run_007_stage2_v23_lowlr_anchor \
         run_008_stage2_v24_lowlr_d1d2; do
  del_with_log "/home/laure/alphaxiang/training_runs/$d"
done

echo ""
echo "=== Phase 4: prune run_002_pikafish_curriculum/ (keep only best.pt) ==="
for f in /home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/*.pt; do
  if [[ "$(basename "$f")" == "best.pt" ]]; then
    KEPT+=("$f")
    echo "  kept: $f (Stage 1 baseline)"
  else
    del_with_log "$f"
  fi
done

echo ""
echo "=== Phase 5: delete old selfplay dirs (v5 will start fresh) ==="
for d in selfplay_runs selfplay_runs_stage2_FAILED_pessimism_collapse \
         selfplay_runs_stage2_v2 selfplay_runs_stage2_v22_ladder \
         selfplay_runs_stage2_v23_lowlr_anchor selfplay_runs_stage2_v24_lowlr_d1d2 \
         selfplay_runs_stage2_v3 selfplay_runs_stage2_v4; do
  del_with_log "/home/laure/alphaxiang/$d"
done

echo ""
echo "=========================================="
echo "TOTAL FREED: $((FREED_KB / 1024)) MB ≈ $((FREED_KB / 1048576)) GB"
echo "=========================================="
echo ""
echo "=== Files KEPT ==="
echo "  /home/laure/alphaxiang/PEAK_step196000_v4_probe2_score63pct.pt  (root backup)"
echo "  /home/laure/alphaxiang/training_runs/run_006_stage2_v4/latest.pt  (= step 196K)"
echo "  /home/laure/alphaxiang/training_runs/run_006_stage2_v4/PEAK_step196000_probe2_score63pct.pt"
echo "  /home/laure/alphaxiang/training_runs/run_006_stage2_v4/snapshots/latest_step196000.pt"
echo "  /home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt  (Stage 1 baseline)"
echo ""
echo "=== Final disk usage ==="
du -sh /home/laure/alphaxiang/training_runs /home/laure/alphaxiang/selfplay_runs* 2>/dev/null
