#!/usr/bin/env python3
"""Build a value-only repair shard from v13_value_child_audit JSON files."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    parts = fen.strip().split()
    while len(parts) < 6:
        if len(parts) in {2, 3}:
            parts.append("-")
        elif len(parts) == 4:
            parts.append("0")
        else:
            parts.append("1")
    return " ".join(parts)


def _value_from_cp(cp: float, scale: float) -> float:
    return float(math.tanh(float(cp) / float(scale)))


def _wdl_from_value(value: float) -> list[float]:
    if value > 0.10:
        return [1.0, 0.0, 0.0]
    if value < -0.10:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _selected_position(record: dict[str, Any], *, min_value_regret_cp: float, min_chosen_regret_cp: float) -> bool:
    teacher = record.get("teacher_best_among_model_candidates") or {}
    value_rank = teacher.get("value_rank")
    policy_rank = teacher.get("policy_rank")
    value_pick = record.get("value_pick") or {}
    chosen_regret = None
    chosen_uci = str((record.get("position") or {}).get("chosen_uci") or "")[:4]
    for row in record.get("candidate_rows") or []:
        if str(row.get("move_uci")) == chosen_uci:
            chosen_regret = float(teacher.get("pika_q_root_pov_cp", 0.0)) - float(row.get("pika_q_root_pov_cp", 0.0))
            break
    if float(value_pick.get("pika_regret_cp", 0.0) or 0.0) >= float(min_value_regret_cp):
        return True
    if chosen_regret is not None and chosen_regret >= float(min_chosen_regret_cp):
        return True
    if value_rank is not None and int(value_rank) > 3 and policy_rank is not None and int(policy_rank) <= 8:
        return True
    return False


def _sample_weight(row: dict[str, Any], record: dict[str, Any], *, base_weight: float, max_weight: float) -> float:
    teacher = record.get("teacher_best_among_model_candidates") or {}
    best_cp = float(teacher.get("pika_q_root_pov_cp", row.get("pika_q_root_pov_cp", 0.0)) or 0.0)
    regret = max(0.0, best_cp - float(row.get("pika_q_root_pov_cp", best_cp) or best_cp))
    tactical = min(float(max_weight), float(base_weight) + regret / 300.0)
    if bool(row.get("is_chosen")):
        tactical *= 1.5
    value_pick = record.get("value_pick") or {}
    if str(row.get("move_uci")) == str(value_pick.get("move_uci")):
        tactical *= 1.5
    return min(float(max_weight), tactical)


def _make_sample(
    record: dict[str, Any],
    row: dict[str, Any],
    *,
    cp_scale: float,
    base_weight: float,
    max_weight: float,
) -> dict[str, Any] | None:
    pos = record.get("position") or {}
    fen = str(pos.get("fen") or "")
    move_uci = str(row.get("move_uci") or "")[:4]
    if not fen or not move_uci:
        return None
    board = Board()
    board.set_fen(_pad_fen(fen))
    raw = int(uci_move_to_internal(move_uci))
    if not bool(board.is_legal(raw)):
        return None
    board.push_legal(raw)
    stm_is_black = bool(int(board.turn()) == 1)
    legal_raw = [int(move) for move in board.legal_moves()]
    if not legal_raw:
        return None
    legal_canonical = [int(canonical_action(move, stm_is_black)) for move in legal_raw]
    placeholder = int(legal_canonical[0])
    child_cp_opp = float(row.get("pika_child_eval_opponent_pov_cp", 0.0) or 0.0)
    value = _value_from_cp(child_cp_opp, cp_scale)
    return {
        "state": board.to_tensor_canonical().to(torch.float32)[0].to(torch.bfloat16).contiguous().clone(),
        "fen": _pad_fen(board.fen()),
        "stm_is_black": stm_is_black,
        "policy_idxs": torch.tensor([placeholder], dtype=torch.int64),
        "policy_probs": torch.tensor([1.0], dtype=torch.float32),
        "chosen_move": placeholder,
        "legal_idxs": torch.tensor(legal_canonical, dtype=torch.int64),
        "z": value,
        "oracle_value": value,
        "wdl_target": _wdl_from_value(value),
        "root_value": value,
        "root_wdl_value": value,
        "sample_weight": _sample_weight(row, record, base_weight=base_weight, max_weight=max_weight),
        "ply": int(pos.get("ply", 0)) + 1,
        "source_game_index": int(pos.get("game_index", -1)),
        "source_arena": str(pos.get("source_arena", "")),
        "source_result": str(pos.get("result", "")),
        "source_move_uci": move_uci,
        "source_policy_rank": int(row.get("policy_rank", 0) or 0),
        "pika_child_eval_opponent_pov_cp": child_cp_opp,
        "pika_q_root_pov_cp": float(row.get("pika_q_root_pov_cp", 0.0) or 0.0),
        "model_child_value_opponent_pov": float(row.get("model_child_value_opponent_pov", 0.0) or 0.0),
    }


def _write_shard(samples: list[dict[str, Any]], path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    policy_offsets = [0]
    legal_offsets = [0]
    policy_idxs: list[torch.Tensor] = []
    policy_probs: list[torch.Tensor] = []
    legal_idxs: list[torch.Tensor] = []
    for sample in samples:
        pi = sample["policy_idxs"].to(torch.int64)
        pp = sample["policy_probs"].float()
        leg = sample["legal_idxs"].to(torch.int64)
        policy_idxs.append(pi)
        policy_probs.append(pp)
        legal_idxs.append(leg)
        policy_offsets.append(policy_offsets[-1] + int(pi.numel()))
        legal_offsets.append(legal_offsets[-1] + int(leg.numel()))
    payload = {
        "state": torch.stack([sample["state"] for sample in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs, dim=0).contiguous(),
        "policy_probs": torch.cat(policy_probs, dim=0).contiguous(),
        "z": torch.tensor([float(sample["z"]) for sample in samples], dtype=torch.float32),
        "oracle_value": torch.tensor([float(sample["oracle_value"]) for sample in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([sample["wdl_target"] for sample in samples], dtype=torch.float32),
        "root_value": torch.tensor([float(sample["root_value"]) for sample in samples], dtype=torch.float32),
        "root_wdl_value": torch.tensor([float(sample["root_wdl_value"]) for sample in samples], dtype=torch.float32),
        "chosen_move": torch.tensor([int(sample["chosen_move"]) for sample in samples], dtype=torch.int64),
        "num_legal_moves": torch.tensor([int(sample["legal_idxs"].numel()) for sample in samples], dtype=torch.int32),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.cat(legal_idxs, dim=0).contiguous(),
        "sample_weight": torch.tensor([float(sample["sample_weight"]) for sample in samples], dtype=torch.float32),
        "ply": torch.tensor([int(sample["ply"]) for sample in samples], dtype=torch.int16),
        "game_id": torch.arange(len(samples), dtype=torch.int64),
        "stm_is_black": torch.tensor([bool(sample["stm_is_black"]) for sample in samples], dtype=torch.bool),
        "is_draw": torch.tensor([abs(float(sample["z"])) <= 0.10 for sample in samples], dtype=torch.bool),
        "termination_code": torch.full((len(samples),), -1, dtype=torch.int8),
        "fens": [str(sample["fen"]) for sample in samples],
        "source_game_index": torch.tensor([int(sample["source_game_index"]) for sample in samples], dtype=torch.int64),
        "source_arena": [str(sample["source_arena"]) for sample in samples],
        "source_result": [str(sample["source_result"]) for sample in samples],
        "source_move_uci": [str(sample["source_move_uci"]) for sample in samples],
        "source_policy_rank": torch.tensor([int(sample["source_policy_rank"]) for sample in samples], dtype=torch.int16),
        "pika_child_eval_opponent_pov_cp": torch.tensor(
            [float(sample["pika_child_eval_opponent_pov_cp"]) for sample in samples], dtype=torch.float32
        ),
        "pika_q_root_pov_cp": torch.tensor([float(sample["pika_q_root_pov_cp"]) for sample in samples], dtype=torch.float32),
        "model_child_value_opponent_pov": torch.tensor(
            [float(sample["model_child_value_opponent_pov"]) for sample in samples], dtype=torch.float32
        ),
        "value_repair_meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-value-regret-cp", type=float, default=300.0)
    parser.add_argument("--min-chosen-regret-cp", type=float, default=300.0)
    parser.add_argument("--max-policy-rank", type=int, default=8)
    parser.add_argument("--cp-scale", type=float, default=500.0)
    parser.add_argument("--base-weight", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=12.0)
    args = parser.parse_args()

    samples: list[dict[str, Any]] = []
    selected_positions = 0
    seen: set[tuple[str, str]] = set()
    for audit_path in [Path(path) for path in args.audit_json]:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if not _selected_position(
                record,
                min_value_regret_cp=float(args.min_value_regret_cp),
                min_chosen_regret_cp=float(args.min_chosen_regret_cp),
            ):
                continue
            selected_positions += 1
            for row in record.get("candidate_rows") or []:
                if int(row.get("policy_rank", 9999) or 9999) > int(args.max_policy_rank):
                    continue
                if "pika_child_eval_opponent_pov_cp" not in row:
                    continue
                sample = _make_sample(
                    record,
                    row,
                    cp_scale=float(args.cp_scale),
                    base_weight=float(args.base_weight),
                    max_weight=float(args.max_weight),
                )
                if sample is None:
                    continue
                key = (sample["fen"], sample["source_move_uci"])
                if key in seen:
                    continue
                samples.append(sample)
                seen.add(key)

    if not samples:
        raise SystemExit("no repair samples selected")

    out = Path(args.output_dir)
    train_dir = out / "train"
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "v13_value_repair_slice",
        "audit_json": [str(Path(path).resolve()) for path in args.audit_json],
        "selected_positions": int(selected_positions),
        "samples": len(samples),
        "min_value_regret_cp": float(args.min_value_regret_cp),
        "min_chosen_regret_cp": float(args.min_chosen_regret_cp),
        "max_policy_rank": int(args.max_policy_rank),
        "cp_scale": float(args.cp_scale),
        "base_weight": float(args.base_weight),
        "max_weight": float(args.max_weight),
    }
    shard = _write_shard(samples, train_dir / "shard_000000.pt", meta)
    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "v13_value_repair_slice",
        "samples": len(samples),
        "selected_positions": int(selected_positions),
        "shards": [shard],
        "quality": "ok",
        "quality_metrics": {"rep_draw_rate": 0.0, "decisive_rate": 100.0, "nocap_draw_rate": 0.0},
        "config": meta,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: {len(samples)} samples from {selected_positions} selected positions -> {train_dir}", flush=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
