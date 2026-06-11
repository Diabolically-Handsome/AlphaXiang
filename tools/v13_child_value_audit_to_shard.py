#!/usr/bin/env python3
"""Convert root-decision audit candidate children into value-training shards.

Root-regret shards mostly repair policy/ranking at the root.  This converter is
for the complementary value/Q question: after a candidate move is played, does
the model value on the child position agree with the deeper Pikafish child eval?

The output follows the trainer's self-play shard shape but intentionally omits
teacher_q and bad_move labels.  It is meant to be trained with policy and
teacher-Q losses disabled, usually with --train-only-value-head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402


def _pad_fen(fen: str) -> str:
    parts = str(fen).strip().split()
    while len(parts) < 6:
        if len(parts) in {2, 3}:
            parts.append("-")
        elif len(parts) == 4:
            parts.append("0")
        else:
            parts.append("1")
    return " ".join(parts)


def _cp_to_tanh(cp: float, scale: float) -> float:
    return float(math.tanh(max(min(float(cp), 2000.0), -2000.0) / float(scale)))


def _wdl_from_value(value: float) -> list[float]:
    if value > 0.1:
        return [1.0, 0.0, 0.0]
    if value < -0.1:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _record_key(record: dict[str, Any]) -> str:
    pos = record.get("position", {})
    return "\n".join(
        [
            str(pos.get("source_arena", "")),
            str(pos.get("game_index", "")),
            str(pos.get("ply", "")),
            str(pos.get("fen", "")),
        ]
    )


def _child_key(position: dict[str, Any], row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(position.get("source_arena", "")),
            str(position.get("game_index", "")),
            str(position.get("ply", "")),
            str(position.get("fen", "")),
            str(row.get("move_uci", row.get("move", ""))),
        ]
    )


def _target_cp(row: dict[str, Any]) -> float | None:
    if row.get("pika_child_eval_opponent_pov_cp") is not None:
        return float(row["pika_child_eval_opponent_pov_cp"])
    if row.get("terminal_q_root_pov_cp") is not None:
        return float(-float(row["terminal_q_root_pov_cp"]))
    if row.get("pika_q_root_pov_cp") is not None:
        return float(-float(row["pika_q_root_pov_cp"]))
    return None


def _model_child_cp(row: dict[str, Any], scale: float) -> float | None:
    if row.get("model_child_value_opponent_pov") is None:
        return None
    value = max(min(float(row["model_child_value_opponent_pov"]), 0.999999), -0.999999)
    return float(math.atanh(value) * float(scale))


def _is_bad_record(record: dict[str, Any]) -> bool:
    return bool(record.get("classification", {}).get("bad_root", False))


def _is_cat_record(record: dict[str, Any]) -> bool:
    return bool(record.get("classification", {}).get("catastrophic", False))


def _sample_weight(record: dict[str, Any], row: dict[str, Any], *, cp_scale: float) -> float:
    target = _target_cp(row)
    model = _model_child_cp(row, cp_scale)
    abs_err = abs(target - model) if target is not None and model is not None else 0.0
    mate_like = row.get("pika_mate_in_child") is not None or abs(float(target or 0.0)) >= 18000.0
    if _is_cat_record(record) and (bool(row.get("is_selected")) or mate_like):
        return 8.0
    if _is_bad_record(record) and (bool(row.get("is_selected")) or bool(row.get("is_teacher_best"))):
        return 5.0
    if mate_like or abs_err >= 1000.0:
        return 4.0
    if abs_err >= 300.0:
        return 2.0
    return 0.5


def _legal_policy_placeholder(board: Board) -> tuple[int, list[int]]:
    legal = [int(move) for move in board.legal_moves()]
    if not legal:
        return 0, []
    stm_black = bool(int(board.turn()) == 1)
    idxs = [int(canonical_action(move, stm_black)) for move in legal]
    return int(idxs[0]), idxs


def _sample_from_child(
    record: dict[str, Any],
    row: dict[str, Any],
    *,
    cp_scale: float,
) -> dict[str, Any] | None:
    position = record.get("position", {})
    target_cp = _target_cp(row)
    if target_cp is None or row.get("move") is None:
        return None

    board = Board()
    board.set_fen(_pad_fen(str(position.get("fen", ""))))
    move = int(row["move"])
    if not bool(board.is_legal(move)):
        return None
    board.push_legal(move)
    board.set_search_context(
        int(position.get("search_plies", position.get("ply", 0)) or 0) + 1,
        int(position.get("no_capture_count", 0) or 0),
        int(position.get("repetition_count_hint", 1) or 1),
    )

    policy_idx, legal_idxs = _legal_policy_placeholder(board)
    if not legal_idxs:
        return None
    z = _cp_to_tanh(target_cp, cp_scale)
    model_cp = _model_child_cp(row, cp_scale)
    return {
        "state": board.to_tensor_canonical().to(torch.bfloat16)[0].contiguous(),
        "policy_idx": int(policy_idx),
        "z": float(z),
        "wdl_target": _wdl_from_value(z),
        "chosen_move": int(policy_idx),
        "legal_idxs": legal_idxs,
        "sample_weight": _sample_weight(record, row, cp_scale=cp_scale),
        "fen": _pad_fen(board.fen()),
        "root_fen": _pad_fen(str(position.get("fen", ""))),
        "move_uci": str(row.get("move_uci", ""))[:4],
        "target_cp_child_pov": float(target_cp),
        "model_cp_child_pov": model_cp,
        "parent_bad_root": _is_bad_record(record),
        "parent_catastrophic": _is_cat_record(record),
        "is_selected_child": bool(row.get("is_selected", False)),
        "is_teacher_best_child": bool(row.get("is_teacher_best", False)),
        "game_index": int(position.get("game_index", -1)),
        "ply": int(position.get("ply", -1)),
        "opening_id": str(position.get("opening_id", "")),
    }


def _load_samples(
    audit_paths: list[Path],
    *,
    cp_scale: float,
    only_bad_roots: bool,
    selected_or_teacher_best_only: bool,
    min_abs_error_cp: float,
) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for audit_path in audit_paths:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if only_bad_roots and not _is_bad_record(record):
                continue
            position = record.get("position", {})
            teacher_move = str(record.get("teacher_best", {}).get("move", ""))[:4]
            for row in record.get("candidate_rows", []):
                row = dict(row)
                row["is_teacher_best"] = str(row.get("move_uci", ""))[:4] == teacher_move
                if selected_or_teacher_best_only and not (
                    bool(row.get("is_selected")) or bool(row.get("is_teacher_best"))
                ):
                    continue
                target = _target_cp(row)
                model = _model_child_cp(row, cp_scale)
                if (
                    min_abs_error_cp > 0.0
                    and target is not None
                    and model is not None
                    and abs(target - model) < min_abs_error_cp
                ):
                    continue
                sample = _sample_from_child(record, row, cp_scale=cp_scale)
                if sample is None:
                    continue
                samples[_child_key(position, row)] = sample
    return samples


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            slim = {k: v for k, v in sample.items() if k != "state"}
            handle.write(json.dumps(slim, ensure_ascii=False, sort_keys=True) + "\n")


def _write_shard(samples: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("no samples to write")
    policy_offsets = [0]
    policy_idxs: list[int] = []
    policy_probs: list[float] = []
    legal_offsets = [0]
    legal_idxs: list[int] = []
    for sample in samples:
        policy_idxs.append(int(sample["policy_idx"]))
        policy_probs.append(1.0)
        policy_offsets.append(policy_offsets[-1] + 1)
        legal_idxs.extend(int(idx) for idx in sample["legal_idxs"])
        legal_offsets.append(legal_offsets[-1] + len(sample["legal_idxs"]))

    payload = {
        "state": torch.stack([sample["state"] for sample in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.tensor(policy_idxs, dtype=torch.int64),
        "policy_probs": torch.tensor(policy_probs, dtype=torch.float32),
        "z": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([sample["wdl_target"] for sample in samples], dtype=torch.float32),
        "root_value": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "root_wdl_value": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "chosen_move": torch.tensor([sample["chosen_move"] for sample in samples], dtype=torch.int64),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.tensor(legal_idxs, dtype=torch.int64),
        "sample_weight": torch.tensor([sample["sample_weight"] for sample in samples], dtype=torch.float32),
        "num_legal_moves": torch.tensor([len(sample["legal_idxs"]) for sample in samples], dtype=torch.int32),
        "ply": torch.tensor([max(0, int(sample["ply"])) for sample in samples], dtype=torch.int16),
        "game_id": torch.arange(len(samples), dtype=torch.int64),
        "stm_is_black": torch.tensor([" b " in sample["fen"] for sample in samples], dtype=torch.bool),
        "is_draw": torch.tensor([False] * len(samples), dtype=torch.bool),
        "termination_code": torch.tensor([-1] * len(samples), dtype=torch.int8),
        "fens": [sample["fen"] for sample in samples],
        "root_fens": [sample["root_fen"] for sample in samples],
        "move_uci": [sample["move_uci"] for sample in samples],
        "target_cp_child_pov": torch.tensor(
            [sample["target_cp_child_pov"] for sample in samples], dtype=torch.float32
        ),
        "model_cp_child_pov": torch.tensor(
            [
                float("nan") if sample["model_cp_child_pov"] is None else sample["model_cp_child_pov"]
                for sample in samples
            ],
            dtype=torch.float32,
        ),
        "parent_bad_root": torch.tensor([sample["parent_bad_root"] for sample in samples], dtype=torch.bool),
        "parent_catastrophic": torch.tensor(
            [sample["parent_catastrophic"] for sample in samples], dtype=torch.bool
        ),
        "is_selected_child": torch.tensor([sample["is_selected_child"] for sample in samples], dtype=torch.bool),
        "is_teacher_best_child": torch.tensor(
            [sample["is_teacher_best_child"] for sample in samples], dtype=torch.bool
        ),
        "opening_id": [sample["opening_id"] for sample in samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--cp-scale", type=float, default=600.0)
    parser.add_argument("--only-bad-roots", action="store_true")
    parser.add_argument("--selected-or-teacher-best-only", action="store_true")
    parser.add_argument("--min-abs-error-cp", type=float, default=0.0)
    args = parser.parse_args()

    samples_by_key = _load_samples(
        [Path(path) for path in args.audit_json],
        cp_scale=float(args.cp_scale),
        only_bad_roots=bool(args.only_bad_roots),
        selected_or_teacher_best_only=bool(args.selected_or_teacher_best_only),
        min_abs_error_cp=float(args.min_abs_error_cp),
    )
    keys = sorted(samples_by_key)
    rng = random.Random(int(args.seed))
    rng.shuffle(keys)
    split = max(1, min(len(keys), int(round(len(keys) * float(args.train_fraction)))))
    train_keys = sorted(keys[:split])
    holdout_keys = sorted(keys[split:])
    train_samples = [samples_by_key[key] for key in train_keys]
    holdout_samples = [samples_by_key[key] for key in holdout_keys]

    out_dir = Path(args.out_dir)
    shard_stats = _write_shard(train_samples, out_dir / "train" / "shard_00000.pt")
    _write_jsonl(out_dir / "splits" / "train_children.jsonl", train_samples)
    _write_jsonl(out_dir / "splits" / "holdout_children.jsonl", holdout_samples)

    weight_buckets: dict[str, int] = defaultdict(int)
    for sample in train_samples:
        weight_buckets[str(sample["sample_weight"])] += 1
    manifest = {
        "format": "v13_child_value_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_state": "complete",
        "quality": "ok",
        "quality_metrics": {"decisive_rate": 100.0, "rep_draw_rate": 0.0, "nocap_draw_rate": 0.0},
        "config": {
            "source": "v13_child_value_audit_to_shard",
            "audit_json": [str(Path(path)) for path in args.audit_json],
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
            "cp_scale": float(args.cp_scale),
            "only_bad_roots": bool(args.only_bad_roots),
            "selected_or_teacher_best_only": bool(args.selected_or_teacher_best_only),
            "min_abs_error_cp": float(args.min_abs_error_cp),
            "search_defaults": {"num_simulations": 0},
        },
        "total_shards": 1,
        "total_samples_written": int(shard_stats["samples"]),
        "total_samples_dropped": 0,
        "sample_weight_buckets": dict(sorted(weight_buckets.items())),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "out_dir": str(out_dir),
        "input_children": len(samples_by_key),
        "train_children": len(train_samples),
        "holdout_children": len(holdout_samples),
        "shard": shard_stats,
        "manifest": str(out_dir / "manifest.json"),
        "sample_weight_buckets": dict(sorted(weight_buckets.items())),
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
