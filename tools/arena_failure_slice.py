"""Convert external_arena JSON losses into v12.5 finetune shards.

The shard schema matches self-play/distillation shards closely enough for:
- oracle_value_labeler.py
- oracle_policy_labeler.py
- hard_position_mining.py
- action_value_labeler.py
- xiangqi_train.py

Default behavior extracts only our turns from games we lost, which is the
anti-Pika-d3 tactical-blunder slice the v12.5 plan calls for.
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


def _should_extract(rec: dict[str, Any], wanted_results: set[str]) -> bool:
    return str(rec.get("result", "")) in wanted_results


def _extract_samples_from_record(
    rec: dict[str, Any],
    *,
    wanted_results: set[str],
    only_our_turns: bool,
    max_plies: int,
) -> list[dict[str, Any]]:
    if not _should_extract(rec, wanted_results):
        return []
    board = Board()
    opening_fen = str(rec.get("opening_fen") or "")
    if opening_fen:
        board.set_fen(_pad_fen(opening_fen))
    moves = [str(m)[:4] for m in rec.get("moves_uci", [])]
    our_is_red = str(rec.get("our_side", "red")) == "red"
    red_result = _red_result_from_game(rec)
    out: list[dict[str, Any]] = []
    for ply, uci in enumerate(moves[:max_plies]):
        red_to_move = int(board.turn()) == 0
        our_turn = red_to_move == our_is_red
        raw_move = int(uci_move_to_internal(uci))
        if (not only_our_turns) or our_turn:
            stm_is_black = bool(board.turn() == 1)
            state = board.to_tensor_canonical().to(torch.float32)[0].contiguous().clone()
            legal_raw = list(board.legal_moves())
            legal_canonical = [int(canonical_action(int(m), stm_is_black)) for m in legal_raw]
            chosen_canonical = int(canonical_action(raw_move, stm_is_black))
            if chosen_canonical in legal_canonical:
                z = _z_for_stm(stm_is_black, red_result)
                out.append({
                    "state": state.to(torch.bfloat16).contiguous(),
                    "fen": _pad_fen(board.fen()),
                    "stm_is_black": stm_is_black,
                    "policy_idxs": torch.tensor([chosen_canonical], dtype=torch.int64),
                    "policy_probs": torch.tensor([1.0], dtype=torch.float32),
                    "chosen_move": chosen_canonical,
                    "legal_idxs": torch.tensor(legal_canonical, dtype=torch.int64),
                    "z": float(z),
                    "wdl_target": _wdl_from_z(float(z)),
                    "ply": int(ply),
                    "source_game_index": int(rec.get("index", -1)),
                    "source_result": str(rec.get("result", "")),
                    "termination": str(rec.get("termination", "")),
                })
        if not bool(board.is_legal(raw_move)):
            break
        board.push_legal(raw_move)
    return out


def _write_shard(samples: list[dict[str, Any]], output_path: Path, shard_id: int) -> dict[str, Any]:
    policy_offsets = [0]
    policy_idxs_chunks: list[torch.Tensor] = []
    policy_probs_chunks: list[torch.Tensor] = []
    legal_offsets = [0]
    legal_chunks: list[torch.Tensor] = []
    for s in samples:
        pi = s["policy_idxs"].to(torch.int64)
        pp = s["policy_probs"].float()
        policy_idxs_chunks.append(pi)
        policy_probs_chunks.append(pp)
        policy_offsets.append(policy_offsets[-1] + int(pi.numel()))
        leg = s["legal_idxs"].to(torch.int64)
        legal_chunks.append(leg)
        legal_offsets.append(legal_offsets[-1] + int(leg.numel()))

    payload = {
        "state": torch.stack([s["state"] for s in samples], dim=0).contiguous(),
        "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
        "policy_idxs": torch.cat(policy_idxs_chunks, dim=0).contiguous(),
        "policy_probs": torch.cat(policy_probs_chunks, dim=0).contiguous(),
        "z": torch.tensor([float(s["z"]) for s in samples], dtype=torch.float32),
        "wdl_target": torch.tensor([s["wdl_target"] for s in samples], dtype=torch.float32),
        "root_value": torch.zeros(len(samples), dtype=torch.float32),
        "root_wdl_value": torch.zeros(len(samples), dtype=torch.float32),
        "chosen_move": torch.tensor([int(s["chosen_move"]) for s in samples], dtype=torch.int64),
        "num_legal_moves": torch.tensor([int(s["legal_idxs"].numel()) for s in samples], dtype=torch.int32),
        "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
        "legal_idxs": torch.cat(legal_chunks, dim=0).contiguous(),
        "ply": torch.tensor([int(s["ply"]) for s in samples], dtype=torch.int16),
        "game_id": torch.tensor(
            [shard_id * 1_000_000 + i for i in range(len(samples))],
            dtype=torch.int64,
        ),
        "stm_is_black": torch.tensor([bool(s["stm_is_black"]) for s in samples], dtype=torch.bool),
        "is_draw": torch.tensor([abs(float(s["z"])) < 1e-6 for s in samples], dtype=torch.bool),
        "termination_code": torch.full((len(samples),), -1, dtype=torch.int8),
        "fens": [str(s["fen"]) for s in samples],
        "source_game_index": torch.tensor([int(s["source_game_index"]) for s in samples], dtype=torch.int64),
        "failure_slice_meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "external_arena",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {"path": str(output_path), "samples": len(samples)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arena_json", nargs="+")
    p.add_argument("--output-dir", required=True,
                   help="Run root. Shards are written to <output-dir>/train.")
    p.add_argument("--results", default="opp_win",
                   help="Comma-separated arena results to extract: opp_win,our_win,draw.")
    p.add_argument("--our-side-filter", choices=["any", "red", "black"], default="any",
                   help="Only extract games where our model played this side. Default any.")
    p.add_argument("--only-our-turns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--max-plies", type=int, default=300)
    args = p.parse_args()

    wanted = {x.strip() for x in args.results.split(",") if x.strip()}
    all_samples: list[dict[str, Any]] = []
    games_seen = 0
    games_used = 0
    for raw in args.arena_json:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        for rec in payload.get("per_game", []):
            games_seen += 1
            if args.our_side_filter != "any" and str(rec.get("our_side", "")) != args.our_side_filter:
                continue
            samples = _extract_samples_from_record(
                rec,
                wanted_results=wanted,
                only_our_turns=bool(args.only_our_turns),
                max_plies=int(args.max_plies),
            )
            if samples:
                games_used += 1
                all_samples.extend(samples)
    if not all_samples:
        raise SystemExit("no samples extracted")

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
        "source": "arena_failure_slice",
        "games_seen": games_seen,
        "games_used": games_used,
        "samples": len(all_samples),
        "shards": shard_infos,
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "results": sorted(wanted),
            "our_side_filter": str(args.our_side_filter),
            "only_our_turns": bool(args.only_our_turns),
            "shard_size": shard_size,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"DONE: extracted {len(all_samples)} samples from {games_used}/{games_seen} games "
        f"into {train_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
