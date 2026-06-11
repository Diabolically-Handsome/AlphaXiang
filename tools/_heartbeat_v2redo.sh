#!/bin/bash
# Robust 10-min heartbeat for the v2redo trainer (run_056). Absolute paths, ps-based liveness.
R=/home/laure/alphaxiang/training_runs/run_056_v2redo_trainer
while true; do
  sleep 600
  alive=$(ps -ef | grep '[x]iangqi_train.py' | grep -c 'run_056_v2redo_trainer')
  snaps=$(ls "$R/snapshots/" 2>/dev/null | grep -c '\.pt')
  doneflag=$(grep -c 'completed normally' "$R/trainer.log" 2>/dev/null)
  last=$(grep 'train step=' "$R/trainer.log" 2>/dev/null | tail -1)
  step=$(echo "$last" | grep -oE 'train step=[0-9]+')
  pol=$(echo "$last" | grep -oE 'policy_loss=[0-9.]+')
  val=$(echo "$last" | grep -oE 'value_loss=[0-9.]+')
  cov=$(echo "$last" | grep -oE 'oracle_cov=[0-9.]+')
  echo "[HB $(date +%H:%M)] alive=$alive done=$doneflag snaps=$snaps | $step $pol $val $cov"
done
