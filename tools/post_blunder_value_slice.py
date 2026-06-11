"""Build a value-focused slice from positions after verified bad moves.

The intent is different from policy imitation.  If MCTS selects a blunder after
search, the policy may already contain the teacher move in top-k; the missing
piece can be value recognition of the refuted child position.  This tool creates
states *after* the bad move so the scalar/WDL heads can learn that danger.
"""
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


def _value_from_cp(cp: float | int | None) -> float:
    if cp is None:
        return 0.0
    return float(math.tanh(float(cp) / 500.0))


def _wdl(value: float) -> list[float]:
    if value > 0.10:
        return [1.0, 0.0, 0.0]
    if value < -0.10:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _reconstruct(opening_fen: str, moves: list[str], ply: int) -> Board | None:
    board = Board()
    if opening_fen:
        board.set_fen(_pad_fen(opening_fen))
    for uci in moves[:ply]:
        raw = int(uci_move_to_internal(str(uci)[:4]))
        if not bool(board.is_legal(raw)):
            return None
        board.push_legal(raw)
    return board


def _make_sample(game: dict[str, Any], row: dict[str, Any], *, include_before: bool) -> dict[str, Any] | None:
    moves = [str(move)[:4] for move in game.get("moves", [])]
    ply = int(row["ply"])
    board = _reconstruct(str(game.get("opening_fen") or ""), moves, ply)
    if board is None:
        return None
    bad_uci = str(row.get("our_move") or "")[:4]
    bad_raw = int(uci_move_to_internal(bad_uci))
    if not include_before:
        if not bool(board.is_legal(bad_raw)):
            return None
        board.push_legal(bad_raw)
    stm_is_black = bool(board.turn() == 1)
    legal_raw = list(board.legal_moves())
    if not legal_raw:
        return None
    legal_canonical = [int(canonical_action(int(move), stm_is_black)) for move in legal_raw]
    chosen_canonical = int(legal_canonical[0])
    if include_before:
        cp_stm = float(row.get("cp_before_our") or 0.0)
    else:
        # After our bad move, side-to-move is the opponent, so invert our-perspective cp.
        cp_stm = -float(row.get("cp_after_our") or 0.0)
    value = _value_from_cp(cp_stm)
    return {
        "state": board.to_tensor_canonical().to(torch.float32)[0].to(torch.bfloat16).contiguous().clone(),
        "fen": _pad_fen(board.fen()),
        "stm_is_black": stm_is_black,
        "policy_idxs": torch.tensor([chosen_canonical], dtype=torch.int64),
        "policy_probs": torch.tensor([1.0], dtype=torch.float32),
        "chosen_move": chosen_canonical,
        "legal_idxs": torch.tensor(legal_canonical, dtype=torch.int64),
        "z": value,
        "wdl_target": _wdl(value),
        "root_value": value,
        "root_wdl_value": value,
        "sample_weight": 4.0 if not include_before else 2.0,
        "ply": ply + (0 if include_before else 1),
        "source_game_index": int(game.get("index", row.get("game_index", -1))),
        "source_depth": str(row.get("depth", "")),
        "source_side": str(row.get("our_side", "")),
        "bad_move_uci": bad_uci,
        "cp_stm": cp_stm,
        "cp_before_our": float(row.get("cp_before_our") or 0.0),
        "cp_after_our": float(row.get("cp_after_our") or 0.0),
        "drop_cp": float(row.get("drop_cp") or 0.0),
        "slice_kind": "before_bad_move" if include_before else "after_bad_move",
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
        "source_depth": [str(sample["source_depth"]) for sample in samples],
        "source_side": [str(sample["source_side"]) for sample in samples],
        "bad_move_uci": [str(sample["bad_move_uci"]) for sample in samples],
        "cp_stm": torch.tensor([float(sample["cp_stm"]) for sample in samples], dtype=torch.float32),
        "cp_before_our": torch.tensor([float(sample["cp_before_our"]) for sample in samples], dtype=torch.float32),
        "cp_after_our": torch.tensor([float(sample["cp_after_our"]) for sample in samples], dtype=torch.float32),
        "drop_cp": torch.tensor([float(sample["drop_cp"]) for sample in samples], dtype=torch.float32),
        "slice_kind": [str(sample["slice_kind"]) for sample in samples],
        "failure_slice_meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-before-cp", type=float, default=-300.0)
    parser.add_argument("--min-drop-cp", type=float, default=250.0)
    parser.add_argument("--min-after-loss-cp", type=float, default=350.0)
    parser.add_argument("--include-before", action="store_true")
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    games = {str(game["gid"]): game for game in analysis.get("loss_games", [])}
    samples: list[dict[str, Any]] = []
    events = 0
    seen: set[tuple[str, int, bool]] = set()
    for row in analysis.get("eval_rows", []):
        before = row.get("cp_before_our")
        after = row.get("cp_after_our")
        drop = row.get("drop_cp")
        if before is None or after is None or drop is None:
            continue
        if float(before) < float(args.min_before_cp):
            continue
        if float(drop) < float(args.min_drop_cp):
            continue
        if float(after) > -float(args.min_after_loss_cp):
            continue
        gid = str(row["gid"])
        game = games.get(gid)
        if game is None:
            continue
        events += 1
        for include_before in ([True, False] if args.include_before else [False]):
            key = (gid, int(row["ply"]), include_before)
            if key in seen:
                continue
            sample = _make_sample(game, row, include_before=include_before)
            if sample is None:
                continue
            samples.append(sample)
            seen.add(key)
    if not samples:
        raise SystemExit("no samples extracted")
    out = Path(args.output_dir)
    train = out / "train"
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "post_blunder_value_slice",
        "analysis_json": str(Path(args.analysis_json).resolve()),
        "min_before_cp": float(args.min_before_cp),
        "min_drop_cp": float(args.min_drop_cp),
        "min_after_loss_cp": float(args.min_after_loss_cp),
        "include_before": bool(args.include_before),
    }
    shard = _write_shard(samples, train / "shard_000000.pt", meta)
    from collections import Counter
    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "post_blunder_value_slice",
        "events": events,
        "samples": len(samples),
        "shards": [shard],
        "quality": "ok",
        "quality_metrics": {"rep_draw_rate": 0.0, "decisive_rate": 100.0, "nocap_draw_rate": 0.0},
        "kind_counts": dict(Counter(sample["slice_kind"] for sample in samples)),
        "depth_counts": dict(Counter(sample["source_depth"] for sample in samples)),
        "side_counts": dict(Counter(sample["source_side"] for sample in samples)),
        "config": meta,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: extracted {len(samples)} samples from {events} events into {train}", flush=True)
    print(json.dumps({k: manifest[k] for k in ("kind_counts", "depth_counts", "side_counts")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

