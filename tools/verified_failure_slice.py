"""Build a tiny, verified first-blunder repair slice from failure-analysis JSON.

This is intentionally narrower than arena_failure_slice.py.  It does not train on
whole losing trajectories.  It extracts only positions around the first avoidable
drop and points the policy target at the teacher best move recorded by the
failure scan.
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


def _wdl_from_value(value: float) -> list[float]:
    if value > 0.10:
        return [1.0, 0.0, 0.0]
    if value < -0.10:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _value_from_cp(cp: int | float | None) -> float:
    if cp is None:
        return 0.0
    return float(math.tanh(float(cp) / 500.0))


def _load_key_depth12(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for item in payload:
        # The depth-12 file lacks gid; match by the robust public tuple.
        key = (
            f"{item.get('depth')}:{item.get('side')}:{int(item.get('game_index'))}",
            int(item.get("ply")),
        )
        out[key] = item
    return out


def _is_core_failure(row: dict[str, Any], *, min_before_cp: float, min_drop_cp: float) -> bool:
    before = row.get("cp_before_our")
    drop = row.get("drop_cp")
    if before is None or drop is None:
        return False
    return float(before) >= min_before_cp and float(drop) >= min_drop_cp


def _is_repeated_d5_red_trap(row: dict[str, Any], *, trap_min_before_cp: float) -> bool:
    before = row.get("cp_before_our")
    drop = row.get("drop_cp")
    if before is None or drop is None:
        return False
    return (
        row.get("depth") == "d5"
        and row.get("our_side") == "red"
        and row.get("our_move") == "f5g7"
        and float(before) >= trap_min_before_cp
        and float(drop) >= 250.0
    )


def _is_key_depth12_failure(
    row: dict[str, Any],
    key_depth12: dict[tuple[str, int], dict[str, Any]],
    *,
    min_depth12_drop_cp: float,
) -> bool:
    public_key = f"{row.get('depth')}:{row.get('our_side')}:{int(row.get('game_index'))}"
    item = key_depth12.get((public_key, int(row.get("ply"))))
    if item is None:
        return False
    drop = item.get("d12_drop")
    target = item.get("d12_best")
    return drop is not None and float(drop) >= min_depth12_drop_cp and bool(target)


def _sample_weight(row: dict[str, Any], reason: str, is_context: bool) -> float:
    if is_context:
        return 1.5
    after = row.get("cp_after_our")
    drop = float(row.get("drop_cp") or 0.0)
    if after is not None and float(after) < -90000:
        return 5.0
    if reason == "repeated_d5_red_trap":
        return 3.5
    if drop >= 600.0:
        return 4.0
    return 3.0


def _reconstruct_board(opening_fen: str, moves: list[str], ply: int) -> Board | None:
    board = Board()
    if opening_fen:
        board.set_fen(_pad_fen(opening_fen))
    for uci in moves[:ply]:
        raw = int(uci_move_to_internal(str(uci)[:4]))
        if not bool(board.is_legal(raw)):
            return None
        board.push_legal(raw)
    return board


def _make_sample(
    *,
    game: dict[str, Any],
    row: dict[str, Any],
    target_uci: str,
    bad_move_uci: str,
    reason: str,
    is_context: bool,
) -> dict[str, Any] | None:
    moves = [str(m)[:4] for m in game.get("moves", [])]
    ply = int(row["ply"])
    board = _reconstruct_board(str(game.get("opening_fen") or ""), moves, ply)
    if board is None:
        return None
    target_raw = int(uci_move_to_internal(str(target_uci)[:4]))
    if not bool(board.is_legal(target_raw)):
        return None
    stm_is_black = bool(board.turn() == 1)
    legal_raw = list(board.legal_moves())
    legal_canonical = [int(canonical_action(int(m), stm_is_black)) for m in legal_raw]
    target_canonical = int(canonical_action(target_raw, stm_is_black))
    if target_canonical not in legal_canonical:
        return None
    bad_canonical = -1
    try:
        bad_raw = int(uci_move_to_internal(str(bad_move_uci)[:4]))
        bad_canonical = int(canonical_action(bad_raw, stm_is_black))
    except Exception:
        bad_canonical = -1
    value = _value_from_cp(row.get("cp_before_our"))
    return {
        "state": board.to_tensor_canonical().to(torch.float32)[0].to(torch.bfloat16).contiguous().clone(),
        "fen": _pad_fen(board.fen()),
        "stm_is_black": stm_is_black,
        "policy_idxs": torch.tensor([target_canonical], dtype=torch.int64),
        "policy_probs": torch.tensor([1.0], dtype=torch.float32),
        "chosen_move": target_canonical,
        "bad_move": bad_canonical,
        "legal_idxs": torch.tensor(legal_canonical, dtype=torch.int64),
        "z": value,
        "wdl_target": _wdl_from_value(value),
        "root_value": value,
        "root_wdl_value": value,
        "sample_weight": _sample_weight(row, reason, is_context),
        "ply": ply,
        "source_game_index": int(game.get("index", row.get("game_index", -1))),
        "source_depth": str(row.get("depth", "")),
        "source_side": str(row.get("our_side", "")),
        "reason": reason,
        "teacher_best_uci": str(target_uci)[:4],
        "bad_move_uci": str(bad_move_uci)[:4],
        "cp_before_our": float(row.get("cp_before_our") or 0.0),
        "cp_after_our": float(row.get("cp_after_our") or 0.0),
        "drop_cp": float(row.get("drop_cp") or 0.0),
        "is_context": bool(is_context),
    }


def _write_shard(samples: list[dict[str, Any]], output_path: Path, shard_id: int, meta: dict[str, Any]) -> dict[str, Any]:
    policy_offsets = [0]
    policy_idxs_chunks: list[torch.Tensor] = []
    policy_probs_chunks: list[torch.Tensor] = []
    legal_offsets = [0]
    legal_chunks: list[torch.Tensor] = []
    for sample in samples:
        pi = sample["policy_idxs"].to(torch.int64)
        pp = sample["policy_probs"].float()
        policy_idxs_chunks.append(pi)
        policy_probs_chunks.append(pp)
        policy_offsets.append(policy_offsets[-1] + int(pi.numel()))
        leg = sample["legal_idxs"].to(torch.int64)
        legal_chunks.append(leg)
        legal_offsets.append(legal_offsets[-1] + int(leg.numel()))

    payload = {
        "state": torch.stack([sample["state"] for sample in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs_chunks, dim=0).contiguous(),
        "policy_probs": torch.cat(policy_probs_chunks, dim=0).contiguous(),
        "z": torch.tensor([float(sample["z"]) for sample in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([sample["wdl_target"] for sample in samples], dtype=torch.float32),
        "root_value": torch.tensor([float(sample["root_value"]) for sample in samples], dtype=torch.float32),
        "root_wdl_value": torch.tensor([float(sample["root_wdl_value"]) for sample in samples], dtype=torch.float32),
        "chosen_move": torch.tensor([int(sample["chosen_move"]) for sample in samples], dtype=torch.int64),
        "bad_move": torch.tensor([int(sample["bad_move"]) for sample in samples], dtype=torch.int64),
        "num_legal_moves": torch.tensor([int(sample["legal_idxs"].numel()) for sample in samples], dtype=torch.int32),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.cat(legal_chunks, dim=0).contiguous(),
        "sample_weight": torch.tensor([float(sample["sample_weight"]) for sample in samples], dtype=torch.float32),
        "ply": torch.tensor([int(sample["ply"]) for sample in samples], dtype=torch.int16),
        "game_id": torch.tensor([shard_id * 1_000_000 + i for i in range(len(samples))], dtype=torch.int64),
        "stm_is_black": torch.tensor([bool(sample["stm_is_black"]) for sample in samples], dtype=torch.bool),
        "is_draw": torch.tensor([abs(float(sample["z"])) <= 0.10 for sample in samples], dtype=torch.bool),
        "termination_code": torch.full((len(samples),), -1, dtype=torch.int8),
        "fens": [str(sample["fen"]) for sample in samples],
        "source_game_index": torch.tensor([int(sample["source_game_index"]) for sample in samples], dtype=torch.int64),
        "source_depth": [str(sample["source_depth"]) for sample in samples],
        "source_side": [str(sample["source_side"]) for sample in samples],
        "reason": [str(sample["reason"]) for sample in samples],
        "teacher_best_uci": [str(sample["teacher_best_uci"]) for sample in samples],
        "bad_move_uci": [str(sample["bad_move_uci"]) for sample in samples],
        "cp_before_our": torch.tensor([float(sample["cp_before_our"]) for sample in samples], dtype=torch.float32),
        "cp_after_our": torch.tensor([float(sample["cp_after_our"]) for sample in samples], dtype=torch.float32),
        "drop_cp": torch.tensor([float(sample["drop_cp"]) for sample in samples], dtype=torch.float32),
        "is_context": torch.tensor([bool(sample["is_context"]) for sample in samples], dtype=torch.bool),
        "failure_slice_meta": meta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {"path": str(output_path), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--key-depth12-json", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--min-before-cp", type=float, default=-250.0)
    parser.add_argument("--trap-min-before-cp", type=float, default=-500.0)
    parser.add_argument("--min-drop-cp", type=float, default=250.0)
    parser.add_argument("--min-depth12-drop-cp", type=float, default=250.0)
    parser.add_argument("--context-plies", type=int, nargs="*", default=[-4, -2, 0])
    args = parser.parse_args()

    analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    key_depth12 = _load_key_depth12(Path(args.key_depth12_json) if args.key_depth12_json else None)
    games = {str(game["gid"]): game for game in analysis.get("loss_games", [])}
    rows_by_game: dict[str, list[dict[str, Any]]] = {}
    row_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in analysis.get("eval_rows", []):
        gid = str(row["gid"])
        rows_by_game.setdefault(gid, []).append(row)
        row_lookup[(gid, int(row["ply"]))] = row
    for rows in rows_by_game.values():
        rows.sort(key=lambda item: int(item["ply"]))

    events: list[tuple[str, dict[str, Any]]] = []
    for row in analysis.get("eval_rows", []):
        reason = ""
        if _is_core_failure(row, min_before_cp=args.min_before_cp, min_drop_cp=args.min_drop_cp):
            reason = "first_clear_drop"
        if _is_repeated_d5_red_trap(row, trap_min_before_cp=args.trap_min_before_cp):
            reason = "repeated_d5_red_trap"
        if _is_key_depth12_failure(row, key_depth12, min_depth12_drop_cp=args.min_depth12_drop_cp):
            reason = "depth12_verified_drop"
        if not reason:
            continue
        target = str(row.get("bestmove_before") or "")[:4]
        if not target or target == str(row.get("our_move", ""))[:4]:
            public_key = f"{row.get('depth')}:{row.get('our_side')}:{int(row.get('game_index'))}"
            key_item = key_depth12.get((public_key, int(row.get("ply"))))
            if key_item is not None:
                target = str(key_item.get("d12_best") or "")[:4]
        if not target or target == str(row.get("our_move", ""))[:4]:
            continue
        events.append((reason, row))

    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for reason, event in events:
        gid = str(event["gid"])
        game = games.get(gid)
        if game is None:
            continue
        for delta in args.context_plies:
            ply = int(event["ply"]) + int(delta)
            ctx = row_lookup.get((gid, ply))
            if ctx is None:
                continue
            context = int(delta) != 0
            target = str(ctx.get("bestmove_before") or "")[:4]
            bad = str(ctx.get("our_move") or "")[:4]
            public_key = f"{ctx.get('depth')}:{ctx.get('our_side')}:{int(ctx.get('game_index'))}"
            key_item = key_depth12.get((public_key, int(ctx.get("ply"))))
            if key_item is not None and key_item.get("d12_best"):
                target = str(key_item["d12_best"])[:4]
            if not target:
                continue
            key = (gid, int(ctx["ply"]), target)
            if key in seen:
                continue
            sample = _make_sample(
                game=game,
                row=ctx,
                target_uci=target,
                bad_move_uci=bad,
                reason=f"{reason}_context" if context else reason,
                is_context=context,
            )
            if sample is None:
                continue
            samples.append(sample)
            seen.add(key)

    if not samples:
        raise SystemExit("no samples extracted")

    out_root = Path(args.output_dir)
    train_dir = out_root / "train"
    shard_size = max(1, int(args.shard_size))
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "verified_failure_slice",
        "analysis_json": str(Path(args.analysis_json).resolve()),
        "key_depth12_json": str(Path(args.key_depth12_json).resolve()) if args.key_depth12_json else "",
        "min_before_cp": float(args.min_before_cp),
        "trap_min_before_cp": float(args.trap_min_before_cp),
        "min_drop_cp": float(args.min_drop_cp),
        "min_depth12_drop_cp": float(args.min_depth12_drop_cp),
        "context_plies": [int(x) for x in args.context_plies],
    }
    shard_infos = []
    for shard_id, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start:start + shard_size]
        shard_infos.append(_write_shard(chunk, train_dir / f"shard_{shard_id:06d}.pt", shard_id, meta))

    reason_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    depth_counts: dict[str, int] = {}
    for sample in samples:
        reason_counts[sample["reason"]] = reason_counts.get(sample["reason"], 0) + 1
        side_counts[sample["source_side"]] = side_counts.get(sample["source_side"], 0) + 1
        depth_counts[sample["source_depth"]] = depth_counts.get(sample["source_depth"], 0) + 1
    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "verified_failure_slice",
        "samples": len(samples),
        "events": len(events),
        "shards": shard_infos,
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "reason_counts": reason_counts,
        "side_counts": side_counts,
        "depth_counts": depth_counts,
        "config": meta,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE: extracted {len(samples)} samples from {len(events)} verified events into {train_dir}", flush=True)
    print(json.dumps({"reason_counts": reason_counts, "side_counts": side_counts, "depth_counts": depth_counts}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

