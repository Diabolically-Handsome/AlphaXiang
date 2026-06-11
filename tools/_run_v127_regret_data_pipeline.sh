#!/bin/bash
# Build v12.7 regret/teacher_q finetune data.
# Requires root guard revalidation JSONs under v127_guard_reval/root_guard.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
CKPT="/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt"
OUT_BASE="/home/laure/alphaxiang/v127_regret_data"
LOG_DIR="$OUT_BASE/logs"

FAILURE_DIR="$OUT_BASE/failure_d4d5"
ROOT_GUARD_DIR="$OUT_BASE/root_guard_events"
EXISTING_SLICE="/home/laure/alphaxiang/v126_day3_d4_slice"
GUARD_REVAL="/home/laure/alphaxiang/v127_guard_reval/root_guard_fixed"

mkdir -p "$OUT_BASE" "$LOG_DIR"
cd "$REPO"

require_glob() {
    local pattern="$1"
    if ! compgen -G "$pattern" >/dev/null; then
        echo "missing required input: $pattern" >&2
        exit 1
    fi
}

label_value_policy() {
    local run_dir="$1"
    local tag="$2"
    "$PY" tools/oracle_value_labeler.py \
        --input-shard-dir "$run_dir/train" \
        --output-shard-dir "$run_dir/train" \
        --depth 15 \
        --workers 8 \
        --threads-per-worker 1 \
        2>&1 | tee "$LOG_DIR/${tag}_oracle_value.log"
    "$PY" tools/oracle_policy_labeler.py \
        --input-shard-dir "$run_dir/train" \
        --output-shard-dir "$run_dir/train" \
        --depth 8 \
        --multipv 5 \
        --adaptive-temperature \
        --legal-smoothing 0.05 \
        --workers 8 \
        --threads-per-worker 1 \
        2>&1 | tee "$LOG_DIR/${tag}_oracle_policy.log"
    "$PY" tools/shard_hygiene_audit.py "$run_dir/train" \
        --json-out "$run_dir/audit.json" \
        --fail-on-dirty \
        2>&1 | tee "$LOG_DIR/${tag}_audit.log"
}

require_glob "/home/laure/alphaxiang/v126_micro_eval/d4/external_arena_*.json"
require_glob "/home/laure/alphaxiang/v126_micro_full_panel_reval/pika_d5/external_arena_*.json"
require_glob "$GUARD_REVAL/pika_d3/external_arena_*.json"
require_glob "$GUARD_REVAL/pika_d4/external_arena_*.json"
require_glob "$GUARD_REVAL/pika_d5/external_arena_*.json"

if [ ! -f "$FAILURE_DIR/manifest.json" ]; then
    "$PY" tools/arena_failure_slice.py \
        /home/laure/alphaxiang/v126_micro_eval/d4/external_arena_*.json \
        /home/laure/alphaxiang/v126_micro_full_panel_reval/pika_d5/external_arena_*.json \
        --output-dir "$FAILURE_DIR" \
        --results opp_win \
        --only-our-turns \
        --shard-size 2048 \
        2>&1 | tee "$LOG_DIR/failure_extract.log"
fi

label_value_policy "$FAILURE_DIR" failure_d4d5
"$PY" tools/hard_position_mining.py \
    --checkpoint "$CKPT" \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --top-percent 20 \
    --heavy-weight 3.0 \
    --light-weight 1.0 \
    --policy-regret-weight 1.0 \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/failure_hard_mining.log"
"$PY" tools/action_value_labeler.py \
    --input-shard-dir "$FAILURE_DIR/train" \
    --output-shard-dir "$FAILURE_DIR/train" \
    --depth 12 \
    --workers 8 \
    --threads-per-worker 1 \
    --oracle-top-k 6 \
    --mcts-top-k 3 \
    --max-candidates 8 \
    --only-hard \
    --min-sample-weight 2.0 \
    --include-chosen \
    --no-skip-already-labeled \
    2>&1 | tee "$LOG_DIR/failure_teacher_q_after_mining.log"
"$PY" tools/shard_hygiene_audit.py "$FAILURE_DIR/train" \
    --json-out "$FAILURE_DIR/audit.json" \
    --fail-on-dirty \
    2>&1 | tee "$LOG_DIR/failure_final_audit.log"

if [ ! -f "$ROOT_GUARD_DIR/manifest.json" ]; then
    "$PY" tools/root_guard_event_slice.py \
        "$GUARD_REVAL"/pika_d3/external_arena_*.json \
        "$GUARD_REVAL"/pika_d4/external_arena_*.json \
        "$GUARD_REVAL"/pika_d5/external_arena_*.json \
        --output-dir "$ROOT_GUARD_DIR" \
        --sample-weight 3.0 \
        --shard-size 2048 \
        2>&1 | tee "$LOG_DIR/root_guard_extract.log"
fi

label_value_policy "$ROOT_GUARD_DIR" root_guard_events
"$PY" tools/action_value_labeler.py \
    --input-shard-dir "$ROOT_GUARD_DIR/train" \
    --output-shard-dir "$ROOT_GUARD_DIR/train" \
    --depth 12 \
    --workers 8 \
    --threads-per-worker 1 \
    --oracle-top-k 6 \
    --mcts-top-k 3 \
    --max-candidates 8 \
    --only-hard \
    --min-sample-weight 2.0 \
    --include-chosen \
    2>&1 | tee "$LOG_DIR/root_guard_teacher_q.log"
"$PY" tools/shard_hygiene_audit.py "$ROOT_GUARD_DIR/train" \
    --json-out "$ROOT_GUARD_DIR/audit.json" \
    --fail-on-dirty \
    2>&1 | tee "$LOG_DIR/root_guard_final_audit.log"
"$PY" tools/shard_hygiene_audit.py "$EXISTING_SLICE/train" \
    --json-out "$OUT_BASE/existing_v126_day3_d4_slice_audit.json" \
    --fail-on-dirty \
    2>&1 | tee "$LOG_DIR/existing_slice_audit.log"

cat > "$OUT_BASE/sources.json" <<JSON
{
  "manifest_state": "complete",
  "created_at": "$(date -Iseconds)",
  "source": "v127_regret_data_pipeline",
  "selfplay_dirs_for_training": [
    "$FAILURE_DIR",
    "$ROOT_GUARD_DIR",
    "$EXISTING_SLICE"
  ],
  "checkpoint": "$CKPT"
}
JSON

echo "v12.7 regret data pipeline DONE: $OUT_BASE"
