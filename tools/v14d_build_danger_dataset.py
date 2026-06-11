#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS = _REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from external_arena import (  # noqa: E402
    _side_to_move_has_check_forced_mate2,
    _side_to_move_has_forcing_check_win,
    _side_to_move_has_mate1,
)
from pikafish_opponent import internal_move_to_uci  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402


DEFAULT_SOURCES = [
    "/home/laure/alphaxiang/v133_p6_fullpika_round2_black_d5_verified_blunders_teacherq_d18",
    "/home/laure/alphaxiang/v133_p6_fullpika_round3_black_d6_verified_blunders_teacherq_d20",
    "/home/laure/alphaxiang/v133_p6_fullpika_round4_black_d6_verified_blunders_teacherq_d20",
]


def _iter_shards(source_dirs: list[Path]):
    for source_dir in source_dirs:
        for pattern in ("train/*.pt", "*.pt"):
            for path in sorted(source_dir.glob(pattern)):
                yield source_dir, path


def _candidate_state_and_labels(
    *,
    fen: str,
    stm_is_black: bool,
    canonical_move: int,
    gap_cp: float,
    is_known_bad: bool,
    positive_gap_cp: float,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
    forcing_plies: int,
) -> tuple[torch.Tensor, torch.Tensor, int, str] | None:
    board = Board()
    board.set_fen(str(fen))
    raw_move = int(canonical_action(int(canonical_move), bool(stm_is_black)))
    legal = {int(move) for move in board.legal_moves()}
    if raw_move not in legal:
        return None

    board.push(raw_move)
    try:
        state_after = board.to_tensor_canonical().to(torch.float32)[0].contiguous()
        mate1 = bool(_side_to_move_has_mate1(
            board,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
        mate2 = bool(_side_to_move_has_check_forced_mate2(
            board,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
        forcing = bool(_side_to_move_has_forcing_check_win(
            board,
            plies_remaining=int(forcing_plies),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
    finally:
        board.pop()

    value_collapse = float(gap_cp) >= float(positive_gap_cp)
    tactical_refuted = bool(is_known_bad or value_collapse or mate1 or mate2 or forcing)
    labels = torch.tensor(
        [
            float(mate1),
            float(mate2),
            float(forcing),
            float(value_collapse),
            float(tactical_refuted),
        ],
        dtype=torch.float32,
    )
    return state_after, labels, raw_move, internal_move_to_uci(raw_move)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dirs = [Path(path) for path in args.source_dirs]
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"source dir not found: {source_dir}")

    states: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    group_ids: list[int] = []
    candidate_moves: list[int] = []
    candidate_raw_moves: list[int] = []
    candidate_uci: list[str] = []
    gaps: list[float] = []
    q_values: list[float] = []
    source_rows: list[dict[str, Any]] = []

    group_id = 0
    skipped = {
        "no_bad_or_best": 0,
        "no_positive": 0,
        "no_negative": 0,
        "illegal_or_failed": 0,
        "context_filtered": 0,
    }
    built_groups = 0

    for source_dir, shard_path in _iter_shards(source_dirs):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(shard, dict):
            continue
        required = {"fens", "stm_is_black", "teacher_q_offsets", "teacher_q_idxs", "teacher_q_values"}
        if not required.issubset(shard):
            continue

        fens = list(shard["fens"])
        stm_is_black = torch.as_tensor(shard["stm_is_black"], dtype=torch.bool)
        offsets = torch.as_tensor(shard["teacher_q_offsets"], dtype=torch.long)
        idxs_all = torch.as_tensor(shard["teacher_q_idxs"], dtype=torch.long)
        vals_all = torch.as_tensor(shard["teacher_q_values"], dtype=torch.float32)
        bad_moves = torch.as_tensor(shard.get("bad_move", torch.full((len(fens),), -1)), dtype=torch.long)
        is_context = torch.as_tensor(shard.get("is_context", torch.zeros((len(fens),), dtype=torch.bool)), dtype=torch.bool)

        for row in range(len(fens)):
            if bool(args.drop_context) and row < int(is_context.numel()) and bool(is_context[row].item()):
                skipped["context_filtered"] += 1
                continue
            start = int(offsets[row].item())
            end = int(offsets[row + 1].item())
            if end - start < 2:
                skipped["no_bad_or_best"] += 1
                continue

            idxs = idxs_all[start:end]
            vals = vals_all[start:end]
            best_pos = int(torch.argmax(vals).item())
            best_idx = int(idxs[best_pos].item())
            best_val = float(vals[best_pos].item())
            bad_idx = int(bad_moves[row].item()) if row < int(bad_moves.numel()) else -1

            candidates: dict[int, tuple[float, bool]] = {}
            candidates[best_idx] = (0.0, False)
            for idx, val in zip(idxs.tolist(), vals.tolist()):
                gap = best_val - float(val)
                known_bad = int(idx) == bad_idx and gap >= float(args.min_bad_gap_cp)
                if known_bad or gap >= float(args.positive_gap_cp):
                    candidates[int(idx)] = (float(gap), True)
                elif gap <= float(args.safe_gap_cp):
                    candidates.setdefault(int(idx), (float(gap), False))

            positives = [(idx, item) for idx, item in candidates.items() if item[1]]
            negatives = [(idx, item) for idx, item in candidates.items() if not item[1]]
            positives = sorted(positives, key=lambda item: item[1][0], reverse=True)[: int(args.max_positive_per_group)]
            negatives = sorted(negatives, key=lambda item: item[1][0])[: int(args.max_negative_per_group)]
            if not positives:
                skipped["no_positive"] += 1
                continue
            if not negatives:
                skipped["no_negative"] += 1
                continue

            group_added = 0
            for idx, (gap, is_pos) in [*positives, *negatives]:
                q_val = best_val - float(gap)
                built = _candidate_state_and_labels(
                    fen=str(fens[row]),
                    stm_is_black=bool(stm_is_black[row].item()),
                    canonical_move=int(idx),
                    gap_cp=float(gap),
                    is_known_bad=bool(is_pos and int(idx) == bad_idx),
                    positive_gap_cp=float(args.positive_gap_cp),
                    max_plies=int(args.max_plies),
                    repeat_limit=int(args.repeat_limit),
                    repeat_min_ply=int(args.repeat_min_ply),
                    no_capture_limit=int(args.no_capture_limit),
                    forcing_plies=int(args.forcing_plies),
                )
                if built is None:
                    skipped["illegal_or_failed"] += 1
                    continue
                state_after, label, raw_move, uci = built
                states.append(state_after.to(torch.bfloat16))
                labels.append(label)
                group_ids.append(group_id)
                candidate_moves.append(int(idx))
                candidate_raw_moves.append(int(raw_move))
                candidate_uci.append(str(uci))
                gaps.append(float(gap))
                q_values.append(float(q_val))
                source_rows.append({
                    "source_dir": str(source_dir),
                    "shard": str(shard_path),
                    "row": int(row),
                    "fen": str(fens[row]),
                    "stm_is_black": bool(stm_is_black[row].item()),
                    "bad_move": int(bad_idx),
                    "teacher_best": int(best_idx),
                    "is_positive": bool(is_pos),
                })
                group_added += 1
            if group_added >= 2:
                group_id += 1
                built_groups += 1

    if not states:
        raise RuntimeError("no danger samples were built")

    payload = {
        "state_after": torch.stack(states, dim=0).contiguous(),
        "labels": torch.stack(labels, dim=0).contiguous(),
        "group_id": torch.tensor(group_ids, dtype=torch.long),
        "candidate_move": torch.tensor(candidate_moves, dtype=torch.long),
        "candidate_raw_move": torch.tensor(candidate_raw_moves, dtype=torch.long),
        "gap_cp": torch.tensor(gaps, dtype=torch.float32),
        "teacher_q": torch.tensor(q_values, dtype=torch.float32),
        "candidate_uci": candidate_uci,
        "source_rows": source_rows,
        "target_names": [
            "opponent_mate1",
            "opponent_mate2",
            "opponent_forcing_check",
            "value_collapse",
            "tactical_refuted",
        ],
        "meta": {
            "sources": [str(path) for path in source_dirs],
            "skipped": skipped,
            "groups": int(built_groups),
            "samples": len(states),
            "args": vars(args),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V14D action-conditioned danger dataset.")
    parser.add_argument("--source-dirs", nargs="*", default=DEFAULT_SOURCES)
    parser.add_argument("--output", default="/home/laure/alphaxiang/v14d_danger_data/danger_dataset.pt")
    parser.add_argument("--drop-context", action="store_true")
    parser.add_argument("--positive-gap-cp", type=float, default=300.0)
    parser.add_argument("--safe-gap-cp", type=float, default=80.0)
    parser.add_argument("--min-bad-gap-cp", type=float, default=80.0)
    parser.add_argument("--max-positive-per-group", type=int, default=4)
    parser.add_argument("--max-negative-per-group", type=int, default=4)
    parser.add_argument("--forcing-plies", type=int, default=5)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    args = parser.parse_args()

    payload = build_dataset(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "output": str(output),
        "samples": int(payload["state_after"].shape[0]),
        "groups": int(payload["meta"]["groups"]),
        "label_positive_counts": {
            name: int(payload["labels"][:, idx].sum().item())
            for idx, name in enumerate(payload["target_names"])
        },
        "skipped": payload["meta"]["skipped"],
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
