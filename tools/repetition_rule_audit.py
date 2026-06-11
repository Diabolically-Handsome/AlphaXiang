"""Audit Xiangqi repetition / perpetual-check / possible long-chase cases.

This is a read-only diagnostic for arena JSON files. It replays each recorded
game with the C++ Board, records repetition terminals, and applies a conservative
Python-side attack heuristic to identify repeated non-check "chase-like" cycles.

The heuristic is intentionally not a rules engine. It is meant to answer:
"Do we have enough repeated chasing in real games to justify implementing the
full long-chase rule / safety filter?"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS = _REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board  # noqa: E402


TERMINAL_ONGOING = -1
TERMINAL_CHECKMATE_OR_STALEMATE = 0
TERMINAL_MAX_PLIES_DRAW = 1
TERMINAL_REPETITION_DRAW = 2
TERMINAL_NO_CAPTURE_DRAW = 3
TERMINAL_PERPETUAL_CHECK_LOSS = 4

TERMINATION_LABELS = {
    TERMINAL_CHECKMATE_OR_STALEMATE: "mate",
    TERMINAL_MAX_PLIES_DRAW: "max",
    TERMINAL_REPETITION_DRAW: "rep",
    TERMINAL_NO_CAPTURE_DRAW: "nocap",
    TERMINAL_PERPETUAL_CHECK_LOSS: "longcheck",
}

PIECE_NAMES = {
    1: "king",
    2: "advisor",
    3: "elephant",
    4: "horse",
    5: "rook",
    6: "cannon",
    7: "pawn",
}


def _sq(x: int, y: int) -> int:
    return y * 9 + x


def _xy(square: int) -> tuple[int, int]:
    return square % 9, square // 9


def _piece_type(piece: int) -> int:
    return abs(int(piece))


def _inside(x: int, y: int) -> bool:
    return 0 <= x < 9 and 0 <= y < 10


def _opponent(piece: int, other: int) -> bool:
    return piece != 0 and other != 0 and ((piece > 0) != (other > 0))


def _target_record(board: Board, square: int) -> dict[str, Any]:
    piece = int(board.piece_at(square))
    x, y = _xy(square)
    return {
        "square": int(square),
        "xy": [int(x), int(y)],
        "piece": int(piece),
        "piece_type": PIECE_NAMES.get(_piece_type(piece), str(_piece_type(piece))),
        "side": "red" if piece > 0 else "black",
    }


def _sliding_targets(board: Board, from_sq: int, piece: int, cannon: bool) -> list[dict[str, Any]]:
    x, y = _xy(from_sq)
    targets: list[dict[str, Any]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        screens = 0
        cx, cy = x + dx, y + dy
        while _inside(cx, cy):
            sq = _sq(cx, cy)
            other = int(board.piece_at(sq))
            if other != 0:
                if not cannon:
                    if _opponent(piece, other):
                        targets.append(_target_record(board, sq))
                    break
                screens += 1
                if screens == 2:
                    if _opponent(piece, other):
                        targets.append(_target_record(board, sq))
                    break
            cx += dx
            cy += dy
    return targets


def _horse_targets(board: Board, from_sq: int, piece: int) -> list[dict[str, Any]]:
    x, y = _xy(from_sq)
    targets: list[dict[str, Any]] = []
    for dx, dy in ((1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)):
        tx, ty = x + dx, y + dy
        if not _inside(tx, ty):
            continue
        leg_x, leg_y = (x + dx // 2, y) if abs(dx) == 2 else (x, y + dy // 2)
        if int(board.piece_at(_sq(leg_x, leg_y))) != 0:
            continue
        other = int(board.piece_at(_sq(tx, ty)))
        if _opponent(piece, other):
            targets.append(_target_record(board, _sq(tx, ty)))
    return targets


def _pawn_targets(board: Board, from_sq: int, piece: int) -> list[dict[str, Any]]:
    x, y = _xy(from_sq)
    red = piece > 0
    dirs: list[tuple[int, int]] = [(0, -1 if red else 1)]
    crossed = y <= 4 if red else y >= 5
    if crossed:
        dirs.extend([(1, 0), (-1, 0)])
    targets: list[dict[str, Any]] = []
    for dx, dy in dirs:
        tx, ty = x + dx, y + dy
        if not _inside(tx, ty):
            continue
        other = int(board.piece_at(_sq(tx, ty)))
        if _opponent(piece, other):
            targets.append(_target_record(board, _sq(tx, ty)))
    return targets


def _moved_piece_attack_targets(board: Board, to_sq: int, piece: int) -> list[dict[str, Any]]:
    piece_t = _piece_type(piece)
    if piece_t == 5:
        return _sliding_targets(board, to_sq, piece, cannon=False)
    if piece_t == 6:
        return _sliding_targets(board, to_sq, piece, cannon=True)
    if piece_t == 4:
        return _horse_targets(board, to_sq, piece)
    if piece_t == 7:
        return _pawn_targets(board, to_sq, piece)
    return []


def _side_name(side: int) -> str:
    return "red" if int(side) == 0 else "black"


def _pad_fen(fen: str) -> str:
    parts = fen.strip().split()
    while len(parts) < 6:
        if len(parts) == 2:
            parts.append("-")
        elif len(parts) == 3:
            parts.append("-")
        elif len(parts) == 4:
            parts.append("0")
        else:
            parts.append("1")
    return " ".join(parts)


def _cycle_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    checking = Counter(r["side"] for r in records if r.get("gives_check"))
    chase = Counter(r["side"] for r in records if r.get("chase_like"))
    chase_targets: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        if not rec.get("chase_like"):
            continue
        for target in rec.get("attack_targets", []):
            if target.get("piece_type") == "king":
                continue
            key = f"{target.get('side')}:{target.get('piece_type')}@{target.get('square')}"
            chase_targets[rec["side"]][key] += 1
    repeated_targets = {
        side: [(target, count) for target, count in counts.items() if count >= 2]
        for side, counts in chase_targets.items()
    }
    return {
        "cycle_plies": len(records),
        "checking_moves_by_side": dict(checking),
        "chase_moves_by_side": dict(chase),
        "repeated_chase_targets_by_side": repeated_targets,
        "moves": [
            {
                "ply": r["ply"],
                "side": r["side"],
                "move": r["move_uci"],
                "piece": r["piece_name"],
                "capture": r["capture"],
                "gives_check": r["gives_check"],
                "chase_like": r["chase_like"],
                "attack_targets": r["attack_targets"],
            }
            for r in records
        ],
    }


def audit_game(
    rec: dict[str, Any],
    *,
    path: str,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> dict[str, Any]:
    board = Board()
    opening = str(rec.get("opening_fen") or "").strip()
    if opening:
        board.set_fen(_pad_fen(opening))

    key_occurrences: dict[int, list[int]] = defaultdict(list)
    key_occurrences[int(board.key())].append(0)
    move_records: list[dict[str, Any]] = []
    repeated_events: list[dict[str, Any]] = []
    invalid_moves: list[dict[str, Any]] = []

    final_term = TERMINAL_ONGOING
    moves = list(rec.get("moves_uci") or [])
    for i, uci in enumerate(moves, start=1):
        move = int(uci_move_to_internal(str(uci)[:4]))
        from_sq = move // 90
        to_sq = move % 90
        side_before = int(board.turn())
        moved_piece = int(board.piece_at(from_sq))
        captured_piece = int(board.piece_at(to_sq))
        legal = bool(board.is_legal(move))
        if not legal:
            invalid_moves.append({"ply": i, "move": str(uci)[:4], "fen": board.fen()})
        board.push(int(move))
        gives_check = bool(board.in_check_turn())
        term = int(board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit))
        current_key = int(board.key())
        key_occurrences[current_key].append(i)
        rep_count = int(board.current_repetition_count())
        attack_targets = []
        if moved_piece != 0 and captured_piece == 0 and not gives_check:
            attack_targets = [
                t for t in _moved_piece_attack_targets(board, to_sq, moved_piece)
                if t.get("piece_type") != "king"
            ]
        move_record = {
            "ply": i,
            "side": _side_name(side_before),
            "move_uci": str(uci)[:4],
            "piece": int(moved_piece),
            "piece_name": PIECE_NAMES.get(_piece_type(moved_piece), str(_piece_type(moved_piece))),
            "capture": bool(captured_piece != 0),
            "captured_piece": int(captured_piece),
            "gives_check": bool(gives_check),
            "chase_like": bool(attack_targets),
            "attack_targets": attack_targets,
            "repetition_count": rep_count,
            "terminal_code": term,
            "terminal_label": TERMINATION_LABELS.get(term, str(term)),
        }
        move_records.append(move_record)
        if len(key_occurrences[current_key]) >= 2:
            repeated_events.append({
                "ply": i,
                "move": str(uci)[:4],
                "repetition_count": rep_count,
                "side_to_move_after": _side_name(int(board.turn())),
                "in_check_after": bool(board.in_check_turn()),
                "terminal_code": term,
                "terminal_label": TERMINATION_LABELS.get(term, str(term)),
            })
        if term != TERMINAL_ONGOING:
            final_term = term
            break

    terminal_label = TERMINATION_LABELS.get(final_term, str(final_term))
    expected_label = str(rec.get("termination") or "")
    terminal_cycle = None
    suspect_long_chase = False
    suspect_long_check_misdraw = False
    if move_records:
        final_key = int(board.key())
        occs = key_occurrences.get(final_key, [])
        if len(occs) >= 2:
            prev_ply = occs[-2]
            cur_ply = occs[-1]
            cycle_records = [r for r in move_records if prev_ply < int(r["ply"]) <= cur_ply]
            terminal_cycle = _cycle_summary(cycle_records)
            check_total = sum(terminal_cycle["checking_moves_by_side"].values())
            chase_total_by_side = terminal_cycle["chase_moves_by_side"]
            repeated_targets = terminal_cycle["repeated_chase_targets_by_side"]
            suspect_long_check_misdraw = (
                final_term == TERMINAL_REPETITION_DRAW and check_total >= 2
            )
            suspect_long_chase = (
                final_term in (TERMINAL_REPETITION_DRAW, TERMINAL_NO_CAPTURE_DRAW)
                and any(count >= 2 for count in chase_total_by_side.values())
                and any(repeated_targets.get(side) for side in repeated_targets)
            )

    return {
        "path": path,
        "game_index": rec.get("index"),
        "our_side": rec.get("our_side"),
        "result": rec.get("result"),
        "expected_termination": expected_label,
        "replayed_termination": terminal_label,
        "plies_recorded": len(moves),
        "plies_replayed": len(move_records),
        "invalid_moves": invalid_moves,
        "repeated_events": repeated_events,
        "terminal_cycle": terminal_cycle,
        "suspect_long_check_misdraw": bool(suspect_long_check_misdraw),
        "suspect_long_chase": bool(suspect_long_chase),
    }


def _iter_paths(patterns: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for p in matches:
            if p not in seen:
                out.append(p)
                seen.add(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Arena JSON paths or glob patterns")
    ap.add_argument("--max-plies", type=int, default=-1, help="Override JSON config; -1 uses each file config.")
    ap.add_argument("--repeat-limit", type=int, default=-1, help="Override JSON config; -1 uses each file config.")
    ap.add_argument("--repeat-min-ply", type=int, default=-1, help="Override JSON config; -1 uses each file config.")
    ap.add_argument("--no-capture-limit", type=int, default=-1, help="Override JSON config; -1 uses each file config.")
    ap.add_argument("--max-examples", type=int, default=12)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    paths = _iter_paths(args.paths)
    summary: dict[str, Any] = {
        "files": len(paths),
        "games": 0,
        "termination_counts_recorded": Counter(),
        "termination_counts_replayed": Counter(),
        "invalid_move_games": 0,
        "repeated_event_games": 0,
        "suspect_long_check_misdraw_games": 0,
        "suspect_long_chase_games": 0,
        "examples": {
            "suspect_long_check_misdraw": [],
            "suspect_long_chase": [],
            "record_replay_mismatch": [],
        },
    }

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        max_plies = int(args.max_plies) if int(args.max_plies) >= 0 else int(config.get("max_plies", 300))
        repeat_limit = int(args.repeat_limit) if int(args.repeat_limit) >= 0 else int(config.get("repeat_limit", 6))
        repeat_min_ply = (
            int(args.repeat_min_ply)
            if int(args.repeat_min_ply) >= 0
            else int(config.get("repeat_min_ply", 30))
        )
        no_capture_limit = (
            int(args.no_capture_limit)
            if int(args.no_capture_limit) >= 0
            else int(config.get("no_capture_limit", 60))
        )
        for rec in payload.get("per_game") or []:
            summary["games"] += 1
            summary["termination_counts_recorded"][str(rec.get("termination") or "")] += 1
            audited = audit_game(
                rec,
                path=path,
                max_plies=max_plies,
                repeat_limit=repeat_limit,
                repeat_min_ply=repeat_min_ply,
                no_capture_limit=no_capture_limit,
            )
            summary["termination_counts_replayed"][audited["replayed_termination"]] += 1
            if audited["invalid_moves"]:
                summary["invalid_move_games"] += 1
            if audited["repeated_events"]:
                summary["repeated_event_games"] += 1
            if audited["suspect_long_check_misdraw"]:
                summary["suspect_long_check_misdraw_games"] += 1
                if len(summary["examples"]["suspect_long_check_misdraw"]) < int(args.max_examples):
                    summary["examples"]["suspect_long_check_misdraw"].append(audited)
            if audited["suspect_long_chase"]:
                summary["suspect_long_chase_games"] += 1
                if len(summary["examples"]["suspect_long_chase"]) < int(args.max_examples):
                    summary["examples"]["suspect_long_chase"].append(audited)
            if audited["expected_termination"] and audited["expected_termination"] != audited["replayed_termination"]:
                if len(summary["examples"]["record_replay_mismatch"]) < int(args.max_examples):
                    summary["examples"]["record_replay_mismatch"].append(audited)

    def convert(obj: Any) -> Any:
        if isinstance(obj, Counter):
            return dict(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    out = convert(summary)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
