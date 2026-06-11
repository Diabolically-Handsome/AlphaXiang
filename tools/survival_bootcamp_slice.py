"""Build a tactical-survival bootcamp slice from losing arena games.

The slice is intentionally narrow: take our turns from games where the candidate
was checkmated, ask Pikafish for the best defensive/practical move, and emit a
high-weight training shard.  This is meant to teach a new v13 trunk basic
survival instincts before trusting long self-play losses or pretty aggregate
loss curves.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402


_CP_SCALE = 500.0


@dataclass
class PositionSample:
    state: torch.Tensor
    fen: str
    stm_is_black: bool
    legal_idxs: list[int]
    played_uci: str
    played_canonical: int
    ply: int
    source_json: str
    source_game_index: int
    our_side: str
    terminal_plies: int


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


def _cp_to_tanh(cp: int) -> float:
    return float(math.tanh(float(cp) / _CP_SCALE))


def _wdl_from_z(z: float) -> list[float]:
    if z > 0.1:
        return [1.0, 0.0, 0.0]
    if z < -0.1:
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _iter_our_turn_positions(
    arena_json: Path,
    *,
    results: set[str],
    last_our_turns: int,
    max_plies: int,
) -> list[PositionSample]:
    payload = json.loads(arena_json.read_text(encoding="utf-8"))
    out: list[PositionSample] = []
    for rec in payload.get("per_game", []):
        if str(rec.get("result", "")) not in results:
            continue
        board = Board()
        board.set_fen(_pad_fen(str(rec.get("opening_fen") or "")))
        our_side = str(rec.get("our_side", "red"))
        our_is_red = our_side == "red"
        moves = [str(m)[:4] for m in rec.get("moves_uci", [])][:max_plies]
        game_samples: list[PositionSample] = []
        for ply, uci in enumerate(moves):
            try:
                raw_move = int(uci_move_to_internal(uci))
            except Exception:
                break
            red_to_move = int(board.turn()) == 0
            our_turn = red_to_move == our_is_red
            if our_turn:
                stm_is_black = bool(board.turn() == 1)
                legal_raw = list(board.legal_moves())
                legal_canonical = [int(canonical_action(int(m), stm_is_black)) for m in legal_raw]
                played_canonical = int(canonical_action(raw_move, stm_is_black))
                if played_canonical in legal_canonical:
                    game_samples.append(
                        PositionSample(
                            state=board.to_tensor_canonical().to(torch.float32)[0].to(torch.bfloat16).contiguous(),
                            fen=_pad_fen(board.fen()),
                            stm_is_black=stm_is_black,
                            legal_idxs=legal_canonical,
                            played_uci=uci,
                            played_canonical=played_canonical,
                            ply=int(ply),
                            source_json=str(arena_json.resolve()),
                            source_game_index=int(rec.get("index", -1)),
                            our_side=our_side,
                            terminal_plies=int(rec.get("plies", len(moves))),
                        )
                    )
            if not bool(board.is_legal(raw_move)):
                break
            board.push_legal(raw_move)
        if last_our_turns > 0:
            game_samples = game_samples[-int(last_our_turns):]
        out.extend(game_samples)
    return out


def _write_shards(
    samples: list[PositionSample],
    labels: dict[int, tuple[int, str, int, int | None]],
    *,
    output_dir: Path,
    shard_size: int,
    sample_weight: float,
    blunder_extra_weight: float,
) -> dict[str, Any]:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    shard_infos: list[dict[str, Any]] = []
    total_written = 0
    dropped = 0
    blunders = 0

    for shard_id, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start:start + shard_size]
        states: list[torch.Tensor] = []
        policy_offsets = [0]
        policy_idxs: list[int] = []
        policy_probs: list[float] = []
        legal_offsets = [0]
        legal_idxs: list[int] = []
        z_values: list[float] = []
        wdl_targets: list[list[float]] = []
        chosen_moves: list[int] = []
        sample_weights: list[float] = []
        plies: list[int] = []
        stm_is_black: list[bool] = []
        num_legal_moves: list[int] = []
        fens: list[str] = []
        source_game_index: list[int] = []
        source_terminal_plies: list[int] = []
        played_uci_list: list[str] = []
        teacher_uci_list: list[str] = []
        teacher_cp_list: list[int] = []
        teacher_mate_list: list[int] = []

        for local_i, sample in enumerate(chunk):
            global_i = start + local_i
            label = labels.get(global_i)
            if label is None:
                dropped += 1
                continue
            teacher_canonical, teacher_uci, cp, mate_in = label
            if teacher_canonical not in sample.legal_idxs:
                dropped += 1
                continue
            is_blunder = teacher_canonical != sample.played_canonical
            if is_blunder:
                blunders += 1
            weight = float(sample_weight) + (float(blunder_extra_weight) if is_blunder else 0.0)
            z = _cp_to_tanh(int(cp))

            states.append(sample.state)
            policy_idxs.append(int(teacher_canonical))
            policy_probs.append(1.0)
            policy_offsets.append(policy_offsets[-1] + 1)
            legal_idxs.extend(int(x) for x in sample.legal_idxs)
            legal_offsets.append(legal_offsets[-1] + len(sample.legal_idxs))
            z_values.append(float(z))
            wdl_targets.append(_wdl_from_z(float(z)))
            chosen_moves.append(int(teacher_canonical))
            sample_weights.append(weight)
            plies.append(int(sample.ply))
            stm_is_black.append(bool(sample.stm_is_black))
            num_legal_moves.append(len(sample.legal_idxs))
            fens.append(str(sample.fen))
            source_game_index.append(int(sample.source_game_index))
            source_terminal_plies.append(int(sample.terminal_plies))
            played_uci_list.append(str(sample.played_uci))
            teacher_uci_list.append(str(teacher_uci))
            teacher_cp_list.append(int(cp))
            teacher_mate_list.append(0 if mate_in is None else int(mate_in))

        if not states:
            continue

        output_path = train_dir / f"shard_{shard_id:06d}.pt"
        payload = {
            "state": torch.stack(states, dim=0).contiguous(),
            "policy_offsets": torch.tensor(policy_offsets, dtype=torch.int64),
            "policy_idxs": torch.tensor(policy_idxs, dtype=torch.int64),
            "policy_probs": torch.tensor(policy_probs, dtype=torch.float32),
            "z": torch.tensor(z_values, dtype=torch.float32),
            "wdl_target": torch.tensor(wdl_targets, dtype=torch.float32),
            "root_value": torch.tensor(z_values, dtype=torch.float32),
            "root_wdl_value": torch.tensor(z_values, dtype=torch.float32),
            "chosen_move": torch.tensor(chosen_moves, dtype=torch.int64),
            "num_legal_moves": torch.tensor(num_legal_moves, dtype=torch.int32),
            "legal_offsets": torch.tensor(legal_offsets, dtype=torch.int64),
            "legal_idxs": torch.tensor(legal_idxs, dtype=torch.int64),
            "ply": torch.tensor(plies, dtype=torch.int16),
            "game_id": torch.tensor([shard_id * 1_000_000 + i for i in range(len(states))], dtype=torch.int64),
            "stm_is_black": torch.tensor(stm_is_black, dtype=torch.bool),
            "is_draw": torch.tensor([abs(z) <= 0.1 for z in z_values], dtype=torch.bool),
            "termination_code": torch.full((len(states),), -1, dtype=torch.int8),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
            "fens": fens,
            "source_game_index": torch.tensor(source_game_index, dtype=torch.int64),
            "source_terminal_plies": torch.tensor(source_terminal_plies, dtype=torch.int16),
            "survival_played_move_uci": played_uci_list,
            "survival_teacher_move_uci": teacher_uci_list,
            "survival_teacher_cp": torch.tensor(teacher_cp_list, dtype=torch.int32),
            "survival_teacher_mate": torch.tensor(teacher_mate_list, dtype=torch.int16),
            "survival_slice_meta": {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "survival_bootcamp_slice",
                "policy_target": "pikafish_bestmove",
            },
        }
        torch.save(payload, output_path)
        total_written += len(states)
        shard_infos.append({"path": str(output_path), "samples": len(states)})

    return {
        "shards": shard_infos,
        "samples_written": total_written,
        "dropped": dropped,
        "blunders": blunders,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arena_json", nargs="+")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--results", default="opp_win")
    p.add_argument("--last-our-turns", type=int, default=16)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--multipv", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--sample-weight", type=float, default=4.0)
    p.add_argument("--blunder-extra-weight", type=float, default=4.0)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    args = p.parse_args()

    wanted = {x.strip() for x in str(args.results).split(",") if x.strip()}
    samples: list[PositionSample] = []
    for raw in args.arena_json:
        samples.extend(
            _iter_our_turn_positions(
                Path(raw),
                results=wanted,
                last_our_turns=int(args.last_our_turns),
                max_plies=int(args.max_plies),
            )
        )
    if not samples:
        raise SystemExit("no survival samples extracted")

    print(
        f"labelling {len(samples)} survival positions with Pikafish depth={args.depth} "
        f"workers={args.workers}",
        flush=True,
    )
    jobs = [
        PikafishJob(index=i, fen=sample.fen, depth=int(args.depth), multipv=int(args.multipv))
        for i, sample in enumerate(samples)
    ]
    labels: dict[int, tuple[int, str, int, int | None]] = {}
    with PikafishPool(
        num_workers=int(args.workers),
        binary_path=args.pikafish_binary,
        threads_per_worker=1,
        hash_mb=int(args.hash_mb),
    ) as pool:
        pool.submit_all(jobs)
        results = pool.collect(len(jobs), timeout_s=max(120.0, 60.0 * len(jobs) / max(1, int(args.workers))))
    for result in results:
        if result.error:
            continue
        try:
            raw_move = int(uci_move_to_internal(str(result.best_move)[:4]))
            sample = samples[int(result.index)]
            canonical = int(canonical_action(raw_move, bool(sample.stm_is_black)))
        except Exception:
            continue
        labels[int(result.index)] = (canonical, str(result.best_move)[:4], int(result.eval_cp), result.mate_in)

    output_dir = Path(args.output_dir)
    written = _write_shards(
        samples,
        labels,
        output_dir=output_dir,
        shard_size=max(1, int(args.shard_size)),
        sample_weight=float(args.sample_weight),
        blunder_extra_weight=float(args.blunder_extra_weight),
    )
    manifest = {
        "manifest_state": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "survival_bootcamp_slice",
        "input_json": [str(Path(x).resolve()) for x in args.arena_json],
        "samples_extracted": len(samples),
        "samples_labelled": len(labels),
        "samples_written": int(written["samples_written"]),
        "dropped": int(written["dropped"]),
        "blunders": int(written["blunders"]),
        "shards": written["shards"],
        "quality": "ok",
        "quality_metrics": {
            "rep_draw_rate": 0.0,
            "decisive_rate": 100.0,
            "nocap_draw_rate": 0.0,
        },
        "config": {
            "results": sorted(wanted),
            "last_our_turns": int(args.last_our_turns),
            "depth": int(args.depth),
            "multipv": int(args.multipv),
            "sample_weight": float(args.sample_weight),
            "blunder_extra_weight": float(args.blunder_extra_weight),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DONE: wrote {manifest['samples_written']} samples "
        f"({manifest['blunders']} played!=teacher) to {output_dir / 'train'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
