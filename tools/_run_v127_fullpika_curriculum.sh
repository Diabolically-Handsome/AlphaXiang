#!/bin/bash
# V12.7 FullPika Curriculum: V12 PEAK + stronger Pikafish teacher + d3->d6 ladder.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
PY="/home/laure/.virtualenvs/AlphaXiang Transformer/bin/python"
V12_PEAK="/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt"

TRAIN_DIR="${V127_FULLPIKA_TRAIN_DIR:-/home/laure/alphaxiang/training_runs/run_020_v127_fullpika_curriculum}"
DATA_ROOT="${V127_FULLPIKA_DATA_ROOT:-/home/laure/alphaxiang/selfplay_runs_v127_fullpika_curriculum}"
EVAL_ROOT="${V127_FULLPIKA_EVAL_ROOT:-/home/laure/alphaxiang/v127_fullpika_curriculum_eval}"
LOG_DIR="$TRAIN_DIR/v127_fullpika_logs"

START_STEP=286000
FINAL_STEP=298000
STEPS_PER_CYCLE=1500
CYCLES_PER_BLOCK=2
SAMPLES_PER_CYCLE=12000
DISTILL_FRACTION=0.30
DISTILL_DEPTH=20
ORACLE_DEPTH=20
POLICY_ORACLE_DEPTH=14
POLICY_ORACLE_MULTIPV=5
POLICY_ORACLE_TEMP=50
DISTILL_WORKERS="${V127_FULLPIKA_DISTILL_WORKERS:-12}"
ORACLE_WORKERS="${V127_FULLPIKA_ORACLE_WORKERS:-8}"
PIKA_HASH_MB="${V127_FULLPIKA_HASH_MB:-256}"
ORACLE_MAX_WAIT_PER_SHARD_S="${V127_FULLPIKA_ORACLE_MAX_WAIT_PER_SHARD_S:-3600}"
BASE_SEED="${V127_FULLPIKA_SEED:-2026051270}"

mkdir -p "$TRAIN_DIR" "$DATA_ROOT" "$EVAL_ROOT" "$LOG_DIR"
cd "$REPO"

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "missing required file: $path" >&2
        exit 1
    fi
}

get_step() {
    "$PY" - "$TRAIN_DIR/latest.pt" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("global_step", 0)))
PY
}

run_arena() {
    local root="$1"
    local key="$2"
    local device="$3"
    local seed="$4"
    local depth="$5"
    local games="$6"
    local out_dir="$root/$key"
    mkdir -p "$out_dir"
    "$PY" tools/external_arena.py \
        --checkpoint "$TRAIN_DIR/latest.pt" \
        --our-sims 1600 \
        --our-c-puct 1.25 --our-q-weight 1.0 --our-q-clip 1.0 \
        --our-temperature-move 0.1 \
        --games "$games" --parallel-games 4 \
        --output-dir "$out_dir" \
        --device "$device" --seed "$seed" \
        --opp-engine pikafish --opp-depth "$depth" \
        2>&1 | tee "$out_dir/run.log"
}

summarize_ladder() {
    local root="$1"
    "$PY" - "$root" <<'PY'
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for depth in (3, 4, 5, 6):
    files = sorted(glob.glob(str(root / f"pika_d{depth}" / "external_arena_*.json")))
    if not files:
        summary[f"d{depth}"] = {"error": "missing_json"}
        continue
    data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    wins = int(data.get("our_wins", 0))
    losses = int(data.get("opp_wins", 0))
    draws = int(data.get("draws", 0))
    total = wins + losses + draws
    score = data.get("score_rate")
    if score is None:
        score = (wins + 0.5 * draws) / total if total else 0.0
    summary[f"d{depth}"] = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "score_rate": float(score),
        "json": files[-1],
    }

(root / "ladder_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
for key in ("d3", "d4", "d5", "d6"):
    item = summary.get(key, {})
    if "error" in item:
        print(f"{key}: {item['error']}")
    else:
        print(
            f"{key}: {item['wins']}-{item['losses']}-{item['draws']}/"
            f"{item['total']} score={item['score_rate'] * 100:.1f}%"
        )
PY
}

gate_quick_ladder() {
    local root="$1"
    "$PY" - "$root" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "ladder_summary.json").read_text(encoding="utf-8"))
d3 = float(summary.get("d3", {}).get("score_rate", 0.0))
d4 = float(summary.get("d4", {}).get("score_rate", 0.0))
if d3 <= 0.125 and d4 <= 0.125:
    print(
        f"QUICK LADDER HALT: d3={d3 * 100:.1f}% and d4={d4 * 100:.1f}% "
        "are both below the V12 anchor band.",
        file=sys.stderr,
    )
    raise SystemExit(20)
PY
}

run_ladder() {
    local label="$1"
    local games="$2"
    local root="$EVAL_ROOT/$label"
    mkdir -p "$root"
    echo "=== ladder $label games=$games root=$root ==="
    (
        run_arena "$root" pika_d3 cuda:0 "$((BASE_SEED + games * 10 + 3))" 3 "$games"
        run_arena "$root" pika_d5 cuda:0 "$((BASE_SEED + games * 10 + 5))" 5 "$games"
    ) > "$root/queue_cuda0.log" 2>&1 &
    local pid0=$!
    (
        run_arena "$root" pika_d4 cuda:1 "$((BASE_SEED + games * 10 + 4))" 4 "$games"
        run_arena "$root" pika_d6 cuda:1 "$((BASE_SEED + games * 10 + 6))" 6 "$games"
    ) > "$root/queue_cuda1.log" 2>&1 &
    local pid1=$!
    wait "$pid0"
    wait "$pid1"
    summarize_ladder "$root" | tee "$root/ladder_summary.txt"
    if [ "$games" -eq 8 ]; then
        gate_quick_ladder "$root"
    fi
}

run_block() {
    local block_idx="$1"
    local block_start="$2"
    local block_end="$3"
    local profile="$4"
    local opp_depth="$5"
    local noise="$6"
    local current_step
    current_step="$(get_step)"

    if [ "$current_step" -ge "$block_end" ]; then
        echo "block $block_idx already complete: current_step=$current_step >= $block_end"
        return 0
    fi
    if [ "$current_step" -lt "$block_start" ]; then
        echo "block $block_idx cannot start: current_step=$current_step < expected $block_start" >&2
        exit 2
    fi

    local remaining=$((block_end - current_step))
    if [ $((remaining % STEPS_PER_CYCLE)) -ne 0 ]; then
        echo "block $block_idx cannot resume cleanly: remaining steps $remaining not divisible by $STEPS_PER_CYCLE" >&2
        exit 2
    fi
    local cycles_left=$((remaining / STEPS_PER_CYCLE))
    if [ "$cycles_left" -le 0 ]; then
        return 0
    fi

    local reset_args=()
    if [ "$block_idx" -eq 1 ] && [ "$current_step" -eq "$START_STEP" ]; then
        reset_args=(--reset-buffer-on-first-cycle)
    fi

    echo "=== block $block_idx profile=$profile depth=$opp_depth noise=$noise cycles=$cycles_left step $current_step->$block_end ==="
    "$PY" tools/stage1_driver.py \
        --training-output-dir "$TRAIN_DIR" \
        --selfplay-root "$DATA_ROOT" \
        --cycles "$cycles_left" \
        --samples-per-cycle "$SAMPLES_PER_CYCLE" \
        --distill-fraction "$DISTILL_FRACTION" \
        --train-steps-per-cycle "$STEPS_PER_CYCLE" \
        --train-lr-schedule-max-steps 400000 \
        --train-snapshot-interval-steps 2000 \
        --distill-depth "$DISTILL_DEPTH" \
        --distill-workers "$DISTILL_WORKERS" \
        --distill-threads-per-worker 1 \
        --distill-hash-mb "$PIKA_HASH_MB" \
        --distill-random-opening-plies 20 \
        --vspika-profile "$profile:16:$opp_depth:$noise:800" \
        --vspika-parallel-games 8 \
        --device cuda:0 \
        --train-device cuda:0 \
        --selfplay-device cuda:1 \
        --oracle-label \
        --oracle-depth "$ORACLE_DEPTH" \
        --oracle-workers "$ORACLE_WORKERS" \
        --oracle-hash-mb "$PIKA_HASH_MB" \
        --oracle-max-wait-per-shard-s "$ORACLE_MAX_WAIT_PER_SHARD_S" \
        --policy-oracle-label \
        --policy-oracle-depth "$POLICY_ORACLE_DEPTH" \
        --policy-oracle-multipv "$POLICY_ORACLE_MULTIPV" \
        --policy-oracle-temperature-cp "$POLICY_ORACLE_TEMP" \
        --policy-oracle-legal-smoothing 0.0 \
        --policy-oracle-alpha 0.5 \
        --hard-mining \
        --hard-mining-top-percent 10 \
        --hard-mining-heavy-weight 3 \
        --hard-mining-policy-regret-weight 1.0 \
        --train-bootstrap-human-floor 0.10 \
        --train-learning-rate 0.0003 \
        --sanity-probe-every 0 \
        --seed "$((BASE_SEED + block_idx * 1000))" \
        "${reset_args[@]}" \
        2>&1 | tee "$LOG_DIR/block_${block_idx}_${profile}.log"

    local after_step
    after_step="$(get_step)"
    if [ "$after_step" -lt "$block_end" ]; then
        echo "block $block_idx stopped early: current_step=$after_step expected >= $block_end" >&2
        exit 3
    fi
    run_ladder "quick_block_${block_idx}_step_${after_step}" 8
}

require_file "$PY"
require_file "$V12_PEAK"

if [ ! -f "$TRAIN_DIR/latest.pt" ]; then
    echo "Seeding $TRAIN_DIR/latest.pt from V12 PEAK"
    cp "$V12_PEAK" "$TRAIN_DIR/latest.pt"
fi

current="$(get_step)"
if [ "$current" -lt "$START_STEP" ]; then
    echo "unexpected checkpoint step $current < $START_STEP" >&2
    exit 2
fi
if [ "$current" -ge "$FINAL_STEP" ]; then
    echo "training already at step $current; running final ladder only"
    run_ladder "final_step_${current}" 50
    exit 0
fi

run_block 1 286000 289000 d3n10 3 0.10
run_block 2 289000 292000 d4n10 4 0.10
run_block 3 292000 295000 d5n05 5 0.05
run_block 4 295000 298000 d6n00 6 0.00

final_step="$(get_step)"
run_ladder "final_step_${final_step}" 50

"$PY" - "$EVAL_ROOT/final_step_${final_step}/ladder_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
d6 = float(summary.get("d6", {}).get("score_rate", 0.0))
if d6 <= 0.05:
    print(f"FINAL NOTE: d6 score={d6 * 100:.1f}% is still near the old V12 floor; freeze V12.7.")
else:
    print(f"FINAL NOTE: d6 score={d6 * 100:.1f}%; review d3/d4 preservation before any d7 run.")
PY

echo "V12.7 FullPika Curriculum DONE: $TRAIN_DIR"
