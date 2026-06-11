#!/bin/bash
# v12.8H data: mine v12.8E d5 failures, then ask a stronger Pikafish
# to refute both oracle moves and the student's own top legal candidates.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
STUDENT_CKPT="${V128H_STUDENT_CKPT:-/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt}"
OUT_BASE="${V128H_DATA_ROOT:-/home/laure/alphaxiang/v128h_fullpika_refutation_data}"
LOG_DIR="$OUT_BASE/logs"
FAILURE_DIR="$OUT_BASE/v128e_d5_losses"
ARENA_GLOB="${V128H_ARENA_GLOB:-/home/laure/alphaxiang/v128_snapshot_smoke/global_strategy_anchor_step297000_quick/pika_d5/external_arena_*.json}"

VALUE_DEPTH="${V128H_VALUE_DEPTH:-16}"
POLICY_DEPTH="${V128H_POLICY_DEPTH:-10}"
POLICY_MULTIPV="${V128H_POLICY_MULTIPV:-8}"
TEACHER_Q_DEPTH="${V128H_TEACHER_Q_DEPTH:-16}"
WORKERS="${V128H_WORKERS:-8}"
THREADS_PER_WORKER="${V128H_THREADS_PER_WORKER:-1}"
HASH_MB="${V128H_HASH_MB:-128}"
MODEL_DEVICE="${V128H_MODEL_DEVICE:-cuda:0}"
MINING_DEVICE="${V128H_MINING_DEVICE:-cuda:0}"
MODEL_TOP_K="${V128H_MODEL_TOP_K:-6}"
ORACLE_TOP_K="${V128H_ORACLE_TOP_K:-8}"
MAX_CANDIDATES="${V128H_MAX_CANDIDATES:-14}"

mkdir -p "$OUT_BASE" "$LOG_DIR"
cd "$REPO"

require_glob() {
    local pattern="$1"
    if ! compgen -G "$pattern" >/dev/null; then
        echo "missing required input: $pattern" >&2
        exit 1
    fi
}

require_glob "$ARENA_GLOB"
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "missing student checkpoint: $STUDENT_CKPT" >&2
    exit 1
fi

if [ ! -f "$FAILURE_DIR/manifest.json" ]; then
    "$PY" tools/arena_failure_slice.py \
        $ARENA_GLOB \
        --output-dir "$FAILURE_DIR" \
        --results opp_win \
        --only-our-turns \
        --shard-size 512 \
        2>&1 | tee "$LOG_DIR/failure_extract.log"
fi

"$PY" tools/oracle_value_labeler.py \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --depth "$VALUE_DEPTH" \
    --workers "$WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --hash-mb "$HASH_MB" \
    2>&1 | tee "$LOG_DIR/oracle_value_d${VALUE_DEPTH}.log"

"$PY" tools/oracle_policy_labeler.py \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --depth "$POLICY_DEPTH" \
    --multipv "$POLICY_MULTIPV" \
    --adaptive-temperature \
    --legal-smoothing 0.05 \
    --workers "$WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --hash-mb "$HASH_MB" \
    2>&1 | tee "$LOG_DIR/oracle_policy_d${POLICY_DEPTH}_mpv${POLICY_MULTIPV}.log"

"$PY" tools/hard_position_mining.py \
    --checkpoint "$STUDENT_CKPT" \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --top-percent 40 \
    --heavy-weight 5.0 \
    --light-weight 1.0 \
    --policy-regret-weight 1.5 \
    --device "$MINING_DEVICE" \
    2>&1 | tee "$LOG_DIR/hard_mining_student.log"

"$PY" tools/action_value_labeler.py \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --depth "$TEACHER_Q_DEPTH" \
    --workers "$WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --hash-mb "$HASH_MB" \
    --oracle-top-k "$ORACLE_TOP_K" \
    --mcts-top-k 1 \
    --candidate-checkpoint "$STUDENT_CKPT" \
    --model-top-k "$MODEL_TOP_K" \
    --model-device "$MODEL_DEVICE" \
    --model-batch-size 128 \
    --max-candidates "$MAX_CANDIDATES" \
    --only-hard \
    --min-sample-weight 2.0 \
    --include-chosen \
    --no-skip-already-labeled \
    2>&1 | tee "$LOG_DIR/fullpika_teacher_q_d${TEACHER_Q_DEPTH}.log"

"$PY" tools/shard_hygiene_audit.py "$FAILURE_DIR/train" \
    --json-out "$FAILURE_DIR/audit.json" \
    --fail-on-dirty \
    2>&1 | tee "$LOG_DIR/final_audit.log"

cat > "$OUT_BASE/sources.json" <<JSON
{
  "manifest_state": "complete",
  "created_at": "$(date -Iseconds)",
  "source": "v128h_fullpika_refutation_data",
  "student_checkpoint": "$STUDENT_CKPT",
  "arena_glob": "$ARENA_GLOB",
  "selfplay_dirs_for_training": [
    "$FAILURE_DIR"
  ],
  "labeling": {
    "value_depth": $VALUE_DEPTH,
    "policy_depth": $POLICY_DEPTH,
    "policy_multipv": $POLICY_MULTIPV,
    "teacher_q_depth": $TEACHER_Q_DEPTH,
    "model_top_k": $MODEL_TOP_K,
    "oracle_top_k": $ORACLE_TOP_K,
    "max_candidates": $MAX_CANDIDATES
  }
}
JSON

echo "v12.8H full-Pika refutation data DONE: $OUT_BASE"
