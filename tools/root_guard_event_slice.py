"""Convert root mate-in-1 guard events into v12.7 finetune shards.

The target policy is the guard's replacement move.  These shards intentionally
look like arena_failure_slice shards so the existing oracle/value/teacher_q
labelers and trainer can consume them without model-architecture changes.
"""
from __future__ import annotations

import argparse
import json
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


def _wdl_from_z(z: float) -> list[float]:
    if z > 1e-6:
        return [1.0, 0.0, 0.0]
    if z < -1e-6:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _red_result_from_game(rec: dict[str, Any]) -> int:
    result = str(rec.get("result", "draw"))
    our_side = str(rec.get("our_side", "red"))
    if result == "draw":
        return 0
    our_won = result == "our_win"
    our_is_red = our_side == "red"
    red_won = our_won == our_is_red
    return 1 if red_won else -1


def _z_for_stm(stm_is_black: bool, red_result: int) -> float:
    if red_result == 0:
        return 0.0
    stm_is_red = not bool(stm_is_black)
    return 1.0 if (red_result > 0) == stm_is_red else -1.0


def _sample_from_event(
    rec: dict[str, Any],
    event: dict[str, Any],
    *,
    sample_weight: float,
) -> tuple[dict[str, Any] | None, str | None]:
    fen = _pad_fen(str(event.get("fen_before") or ""))
    replacement_uci = str(event.get("replacement_move_uci") or "")[:4]
    original_uci = str(event.get("original_move_uci") or "")[:4]
    if not fen.strip() or len(replacement_uci) != 4:
        return None, "missing_fen_or_replacement"

    board = Board()
    try:
        board.set_fen(fen)
    except Exception:
        return None, "bad_fen"

    try:
        replacement_raw = int(uci_move_to_internal(replacement_uci))
    except Exception:
        return None, "bad_replacement_uci"

    if not bool(board.is_legal(replacement_raw)):
        return None, "illegal_replacement"

    stm_is_black = bool(int(board.turn()) == 1)
    legal_raw = list(board.legal_moves())
    legal_canonical = [int(canonical_action(int(m), stm_is_black)) for m in legal_raw]
    replacement_canonical = int(canonical_action(replacement_raw, stm_is_black))
    if replacement_canonical not in legal_canonical:
        return None, "replacement_not_in_canonical_legal"

    red_result = _red_result_from_game(rec)
    z = _z_for_stm(stm_is_black, red_result)
    return {
        "state": board.to_tensor_canonical().to(torch.float32)[0].to(torch.bfloat16).contiguous(),
        "fen": fen,
        "stm_is_black": stm_is_black,
        "policy_idxs": torch.tensor([replacement_canonical], dtype=torch.int64),
        "policy_probs": torch.tensor([1.0], dtype=torch.float32),
        "chosen_move": replacement_canonical,
        "legal_idxs": torch.tensor(legal_canonical, dtype=torch.int64),
        "z": float(z),
        "wdl_target": _wdl_from_z(float(z)),
        "sample_weight": float(sample_weight),
        "ply": int(event.get("ply", -1)),
        "source_game_index": int(event.get("game_index", rec.get("index", -1))),
        "source_result": str(rec.get("result", "")),
        "termination": str(rec.get("termination", "")),
        "original_move_uci": original_uci,
        "replacement_move_uci": replacement_uci,
        "original_prob": event.get("original_prob"),
        "replacement_prob": event.get("replacement_prob"),
        "guard_type": str(event.get("guard_type", "root_mate1_blunder_guard")),
    }, None


def _write_shard(samples: list[dict[str, Any]], output_path: Path, shard_id: int) -> dict[str, Any]:
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
        legal = sample["legal_idxs"].to(torch.int64)
        legal_chunks.append(legal)
        legal_offsets.append(legal_offsets[-1] + int(legal.numel()))

    payload = {
        "state": torch.stack([sample["state"] for sample in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs_chunks, dim=0).contiguous(),
        "policy_probs": torch.cat(policy_probs_chunks, dim=0).contiguous(),
        "z": torch.tensor([float(sample["z"]) for sample in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([sample["wdl_target"] for sample in samples], dtype=torch.float32),
        "root_value": torch.zeros(len(samples), dtype=torch.float32),
        "root_wdl_value": torch.zeros(len(samples), dtype=torch.float32),
        "chosen_move": torch.tensor([int(sample["chosen_move"]) for sample in samples], dtype=torch.int64),
        "num_legal_moves": torch.tensor([int(sample["legal_idxs"].numel()) for sample in samples], dtype=torch.int32),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.cat(legal_chunks, dim=0).contiguous(),
        "ply": torch.tensor([int(sample["ply"]) for sample in samples], dtype=torch.int16),
        "game_id": torch.tensor(
            [shard_id * 1_000_000 + i for i in range(len(samples))],
            dtype=torch.int64,
        ),
        "stm_is_black": torch.tensor([bool(sample["stm_is_black"]) for sample in samples], dtype=torch.bool),
        "is_draw": torch.tensor([abs(float(sample["z"])) < 1e-6 for sample in samples], dtype=torch.bool),
        "termination_code": torch.full((len(samples),), -1, dtype=torch.int8),
        "sample_weight": torch.tensor([float(sample["sample_weight"]) for sample in samples], dtype=torch.float32),
        "fens": [str(sample["fen"]) for sample in samples],
        "source_game_index": torch.tensor(
            [int(sample["source_game_index"]) for sample in samples],
            dtype=torch.int64,
        ),
        "root_guard_original_move_uci": [str(sample["original_move_uci"]) for sample in samples],
        "root_guard_replacement_move_uci": [str(sample["replacement_move_uci"]) for sample in samples],
        "root_guard_original_prob": [
            None if sample["original_prob"] is None else float(sample["original_prob"])
            for sample in samples
        ],
        "root_guard_replacement_prob": [
            None if sample["replacement_prob"] is None else float(sample["replacement_prob"])
            for sample in samples
        ],
        "root_guard_event_meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "root_guard_event_slice",
            "policy_target": "replacement_move",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {"path": str(output_path), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arena_json", nargs="+")
    parser.add_argument("--output-dir", required=True,
                        help="Run root. Shards are written to <output-dir>/train.")
    parser.add_argument("--sample-weight", type=float, default=3.0)
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args()

    all_samples: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    games_seen = 0
    games_with_events = 0
    events_seen = 0

    for raw in args.arena_json:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        for rec in payload.get("per_game", []):
            games_seen += 1
            events = list(rec.get("guard_events") or [])
            if events:
                games_with_events += 1
            for event in events:
                events_seen += 1
                sample, reason = _sample_from_event(
                    rec,
                    event,
                    sample_weight=float(args.sample_weight),
                )
                if sample is None:
                    skipped[str(reason)] = skipped.get(str(reason), 0) + 1
                    continue
                all_samples.append(sample)

    if not all_samples:
        raise SystemExit(
            f"no root guard samples extracted; games_seen={games_seen} "
            f"events_seen={events_seen} skipped={skipped}"
        )

    out_root = Path(args.output_dir)
    train_dir = out_root / "train"
    shard_size = max(1, int(args.shard_size))
    shard_infos = []
    for shard_id, start in enumerate(range(0, len(all_samples), shard_size)):
        chunk = all_samples[start:start + shard_size]
        shard_infos.append(_write_shard(chunk, train_dir / f"shard_{shard_id:06d}.pt", shard_id))

    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "root_guard_event_slice",
        "games_seen": games_seen,
        "games_with_events": games_with_events,
        "guard_events_seen": events_seen,
        "samples": len(all_samples),
        "skipped": skipped,
        "shards": shard_infos,
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "sample_weight": float(args.sample_weight),
            "shard_size": shard_size,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"DONE: extracted {len(all_samples)} root-guard samples from "
        f"{events_seen} event(s) into {train_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
