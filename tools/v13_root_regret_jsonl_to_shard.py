#!/usr/bin/env python3
"""Convert V13 root-regret JSONL rows into trainer-readable tensor shards.

The output is a tiny self-play-style run directory:

  out_dir/
    manifest.json
    train/shard_00000.pt
    splits/train_roots.jsonl
    splits/holdout_roots.jsonl

The shard carries teacher_q candidates, bad_move labels, legal masks, and
sample_weight.  Ordinary policy/value losses can be disabled by the training
runner; the required self-play fields are still populated for compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
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

from pikafish_opponent import uci_move_to_internal  # noqa: E402
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


def _root_key(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("audit_json", "")),
            str(row.get("fen", "")),
            str(row.get("game_index", "")),
            str(row.get("ply", "")),
            str(row.get("selected_move", "")),
        ]
    )


def _load_grouped(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                grouped[_root_key(row)].append(row)
    return grouped


def _stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _cp_to_tanh(cp: float) -> float:
    return float(math.tanh(max(min(float(cp), 2000.0), -2000.0) / 600.0))


def _selected_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_selected"))), rows[0])


def _teacher_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_teacher_best"))), min(rows, key=lambda row: float(row.get("regret_cp", 0.0))))


def _is_catastrophic(selected: dict[str, Any]) -> bool:
    regret = float(selected.get("regret_cp", 0.0) or 0.0)
    return bool(
        regret >= 150.0
        and (
            regret >= 1000.0
            or selected.get("pika_mate_in_child") is not None
        )
    )


def _sample_weight(selected: dict[str, Any]) -> float:
    regret = float(selected.get("regret_cp", 0.0) or 0.0)
    if regret >= 150.0 and _is_catastrophic(selected):
        return 8.0
    if regret >= 150.0:
        return 5.0
    if regret >= 80.0:
        return 2.0
    return 0.5


def _selected_regret_for_group(rows: list[dict[str, Any]]) -> float:
    selected = _selected_row(rows)
    return float(selected.get("regret_cp", 0.0) or 0.0)


def _board_from_row(row: dict[str, Any]) -> Board:
    board = Board()
    board.set_fen(_pad_fen(str(row.get("fen", ""))))
    board.set_search_context(
        int(row.get("search_plies", row.get("ply", 0)) or 0),
        int(row.get("no_capture_count", 0) or 0),
        int(row.get("repetition_count_hint", 1) or 1),
    )
    return board


def _canonical_move(board: Board, move_uci: str) -> int | None:
    try:
        raw = int(uci_move_to_internal(str(move_uci)[:4]))
    except Exception:
        return None
    if not bool(board.is_legal(raw)):
        return None
    stm_black = bool(int(board.turn()) == 1)
    return int(canonical_action(raw, stm_black))


def _root_to_sample(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _selected_row(rows)
    teacher = _teacher_best_row(rows)
    board = _board_from_row(selected)
    stm_black = bool(int(board.turn()) == 1)

    teacher_q_by_idx: dict[int, float] = {}
    for row in rows:
        idx = _canonical_move(board, str(row.get("candidate_move", "")))
        if idx is None:
            continue
        value = float(
            row.get(
                "child_d20_score_cp",
                row.get("teacher_child_q_root_pov_cp", row.get("child_d16_score_cp", 0.0)),
            )
        )
        if idx not in teacher_q_by_idx or value > teacher_q_by_idx[idx]:
            teacher_q_by_idx[idx] = value
    if len(teacher_q_by_idx) < 2:
        return None

    teacher_idx = _canonical_move(board, str(teacher.get("candidate_move", "")))
    if teacher_idx is None:
        teacher_idx = max(teacher_q_by_idx, key=lambda idx: teacher_q_by_idx[idx])
    selected_idx = _canonical_move(board, str(selected.get("candidate_move", selected.get("selected_move", ""))))
    bad_idx = -1
    if selected_idx is not None and float(selected.get("regret_cp", 0.0) or 0.0) >= 150.0:
        bad_idx = int(selected_idx)

    legal_idxs = [int(canonical_action(int(move), stm_black)) for move in board.legal_moves()]
    teacher_q_items = sorted(teacher_q_by_idx.items(), key=lambda item: item[0])
    teacher_cp = float(teacher_q_by_idx.get(teacher_idx, 0.0))
    z = _cp_to_tanh(teacher_cp)
    if z > 0.1:
        wdl = [1.0, 0.0, 0.0]
    elif z < -0.1:
        wdl = [0.0, 0.0, 1.0]
    else:
        wdl = [0.0, 1.0, 0.0]

    return {
        "state": board.to_tensor_canonical().to(torch.bfloat16)[0].contiguous(),
        "policy_idx": int(teacher_idx),
        "policy_prob": 1.0,
        "z": z,
        "oracle_value": z,
        "oracle_value_cp": teacher_cp,
        "wdl_target": wdl,
        "chosen_move": int(teacher_idx),
        "legal_idxs": legal_idxs,
        "teacher_q_idxs": [int(idx) for idx, _value in teacher_q_items],
        "teacher_q_values": [float(value) for _idx, value in teacher_q_items],
        "bad_move": int(bad_idx),
        "sample_weight": _sample_weight(selected),
        "fen": _pad_fen(str(selected.get("fen", ""))),
        "selected_move_uci": str(selected.get("candidate_move", selected.get("selected_move", "")))[:4],
        "teacher_best_move_uci": str(teacher.get("candidate_move", ""))[:4],
        "selected_regret_cp": float(selected.get("regret_cp", 0.0) or 0.0),
        "is_bad_root": bool(float(selected.get("regret_cp", 0.0) or 0.0) >= 150.0),
        "is_catastrophic_root": _is_catastrophic(selected),
        "pika_label_root_depth": int(selected.get("pika_label_root_depth") or 0),
        "pika_label_child_depth": int(selected.get("pika_label_child_depth") or 0),
    }


def _write_jsonl(path: Path, grouped: dict[str, list[dict[str, Any]]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in keys:
            for row in grouped.get(key, []):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_shard(samples: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("no samples to write")

    policy_offsets = [0]
    policy_idxs: list[int] = []
    policy_probs: list[float] = []
    legal_offsets = [0]
    legal_idxs: list[int] = []
    teacher_q_offsets = [0]
    teacher_q_idxs: list[int] = []
    teacher_q_values: list[float] = []

    for sample in samples:
        policy_idxs.append(int(sample["policy_idx"]))
        policy_probs.append(float(sample["policy_prob"]))
        policy_offsets.append(policy_offsets[-1] + 1)
        legal_idxs.extend(int(idx) for idx in sample["legal_idxs"])
        legal_offsets.append(legal_offsets[-1] + len(sample["legal_idxs"]))
        teacher_q_idxs.extend(int(idx) for idx in sample["teacher_q_idxs"])
        teacher_q_values.extend(float(value) for value in sample["teacher_q_values"])
        teacher_q_offsets.append(teacher_q_offsets[-1] + len(sample["teacher_q_idxs"]))

    payload = {
        "state": torch.stack([sample["state"] for sample in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.tensor(policy_idxs, dtype=torch.int64),
        "policy_probs": torch.tensor(policy_probs, dtype=torch.float32),
        "z": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "oracle_value": torch.tensor([sample["oracle_value"] for sample in samples], dtype=torch.float32),
        "oracle_value_cp": torch.tensor([sample["oracle_value_cp"] for sample in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([sample["wdl_target"] for sample in samples], dtype=torch.float32),
        "root_value": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "root_wdl_value": torch.tensor([sample["z"] for sample in samples], dtype=torch.float32),
        "chosen_move": torch.tensor([sample["chosen_move"] for sample in samples], dtype=torch.int64),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.tensor(legal_idxs, dtype=torch.int64),
        "teacher_q_offsets": torch.tensor(teacher_q_offsets, dtype=torch.int64),
        "teacher_q_idxs": torch.tensor(teacher_q_idxs, dtype=torch.int64),
        "teacher_q_values": torch.tensor(teacher_q_values, dtype=torch.float32),
        "bad_move": torch.tensor([sample["bad_move"] for sample in samples], dtype=torch.int64),
        "sample_weight": torch.tensor([sample["sample_weight"] for sample in samples], dtype=torch.float32),
        "num_legal_moves": torch.tensor([len(sample["legal_idxs"]) for sample in samples], dtype=torch.int32),
        "ply": torch.zeros(len(samples), dtype=torch.int16),
        "game_id": torch.arange(len(samples), dtype=torch.int64),
        "stm_is_black": torch.tensor([" b " in sample["fen"] for sample in samples], dtype=torch.bool),
        "is_draw": torch.tensor([False] * len(samples), dtype=torch.bool),
        "termination_code": torch.tensor([-1] * len(samples), dtype=torch.int8),
        "fens": [sample["fen"] for sample in samples],
        "selected_move_uci": [sample["selected_move_uci"] for sample in samples],
        "teacher_best_move_uci": [sample["teacher_best_move_uci"] for sample in samples],
        "selected_regret_cp": torch.tensor([sample["selected_regret_cp"] for sample in samples], dtype=torch.float32),
        "is_bad_root": torch.tensor([sample["is_bad_root"] for sample in samples], dtype=torch.bool),
        "is_catastrophic_root": torch.tensor([sample["is_catastrophic_root"] for sample in samples], dtype=torch.bool),
        "pika_label_root_depth": torch.tensor([sample["pika_label_root_depth"] for sample in samples], dtype=torch.int16),
        "pika_label_child_depth": torch.tensor([sample["pika_label_child_depth"] for sample in samples], dtype=torch.int16),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "samples": len(samples),
        "bad_roots": int(sum(1 for sample in samples if sample["is_bad_root"])),
        "catastrophic_roots": int(sum(1 for sample in samples if sample["is_catastrophic_root"])),
        "oracle_value_samples": int(sum(1 for sample in samples if math.isfinite(float(sample["oracle_value"])))),
        "oracle_value_coverage": float(
            sum(1 for sample in samples if math.isfinite(float(sample["oracle_value"]))) / max(1, len(samples))
        ),
        "min_pika_root_depth": int(min((sample["pika_label_root_depth"] for sample in samples), default=0)),
        "min_pika_child_depth": int(min((sample["pika_label_child_depth"] for sample in samples), default=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_regret_jsonl", nargs="+")
    parser.add_argument("--exact-jsonl", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--max-roots", type=int, default=0)
    args = parser.parse_args()

    grouped = _load_grouped([Path(path) for path in args.root_regret_jsonl])
    keys = sorted(grouped, key=lambda key: (_selected_regret_for_group(grouped[key]) < 150.0, -_selected_regret_for_group(grouped[key]), _stable_hash_int(key)))
    if int(args.max_roots) > 0:
        keys = keys[: int(args.max_roots)]
    bad_keys = [key for key in keys if _selected_regret_for_group(grouped[key]) >= 150.0]
    non_bad_keys = [key for key in keys if _selected_regret_for_group(grouped[key]) < 150.0]
    rng = random.Random(int(args.seed))
    rng.shuffle(non_bad_keys)
    target_train = max(len(bad_keys), min(len(keys), int(round(len(keys) * float(args.train_fraction)))))
    extra_non_bad = max(0, target_train - len(bad_keys))
    train_keys = sorted(bad_keys + non_bad_keys[:extra_non_bad])
    holdout_keys = sorted(non_bad_keys[extra_non_bad:])

    train_samples: list[dict[str, Any]] = []
    dropped = 0
    for key in train_keys:
        sample = _root_to_sample(grouped[key])
        if sample is None:
            dropped += 1
            continue
        train_samples.append(sample)

    out_dir = Path(args.out_dir)
    shard_stats = _write_shard(train_samples, out_dir / "train" / "shard_00000.pt")
    _write_jsonl(out_dir / "splits" / "train_roots.jsonl", grouped, train_keys)
    _write_jsonl(out_dir / "splits" / "holdout_roots.jsonl", grouped, holdout_keys)

    if args.exact_jsonl:
        exact_grouped = _load_grouped([Path(path) for path in args.exact_jsonl])
        exact_keys = sorted(exact_grouped)
        _write_jsonl(out_dir / "splits" / "exact112_roots.jsonl", exact_grouped, exact_keys)
        with (out_dir / "splits" / "holdout_plus_exact112.jsonl").open("w", encoding="utf-8") as out:
            for path in (out_dir / "splits" / "holdout_roots.jsonl", out_dir / "splits" / "exact112_roots.jsonl"):
                if path.exists():
                    out.write(path.read_text(encoding="utf-8"))

    manifest = {
        "format": "v13_root_regret_teacher_q_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_state": "complete",
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "source": "v13.4_root_ranking_repair",
            "root_regret_jsonl": [str(Path(path)) for path in args.root_regret_jsonl],
            "exact_jsonl": [str(Path(path)) for path in args.exact_jsonl],
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
            "weights": {
                "catastrophic": 8.0,
                "bad": 5.0,
                "near_bad": 2.0,
                "clean": 0.5,
            },
            "search_defaults": {"num_simulations": 0},
        },
        "total_shards": 1,
        "total_samples_written": int(shard_stats["samples"]),
        "total_samples_dropped": int(dropped),
        "oracle_value_coverage": float(shard_stats["oracle_value_coverage"]),
        "oracle_value_samples": int(shard_stats["oracle_value_samples"]),
        "fullpika_depths": {
            "min_pika_root_depth": int(shard_stats["min_pika_root_depth"]),
            "min_pika_child_depth": int(shard_stats["min_pika_child_depth"]),
        },
        "fullpika_ok": bool(
            int(shard_stats["min_pika_root_depth"]) >= 20
            and int(shard_stats["min_pika_child_depth"]) >= 20
            and float(shard_stats["oracle_value_coverage"]) >= 1.0
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "input_roots": len(grouped),
        "selected_roots": len(keys),
        "train_roots": len(train_keys),
        "holdout_roots": len(holdout_keys),
        "dropped_train_roots": int(dropped),
        "shard": shard_stats,
        "manifest": str(out_dir / "manifest.json"),
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
