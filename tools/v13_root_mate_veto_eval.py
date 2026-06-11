#!/usr/bin/env python3
"""Offline evaluation for narrow root mate/horizon veto guards.

This is a frozen-checkpoint diagnostic: it does not run MCTS or train anything.
It replays the audited V13 root decisions and asks whether the existing
symbolic root guards would have vetoed the selected move using only V13 root
top-K alternatives.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from external_arena import (  # noqa: E402
    _move_allows_opponent_check_forced_mate2,
    _move_allows_opponent_forcing_check_win,
    _move_allows_opponent_mate1,
)
from pikafish_opponent import internal_move_to_uci, uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board  # noqa: E402


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


def _load_groups(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source_jsonl"] = str(path)
                    groups[_root_key(row)].append(row)
    return groups


def _load_trajectory_classes(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for record in payload.get("records", []) or []:
        key = str((record.get("root") or {}).get("key", ""))
        if key:
            out[key] = str(record.get("classification", "unknown"))
    return out


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _rank(value: Any, default: int = 999) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("is_selected"))), rows[0])


def _board_from_row(row: dict[str, Any]) -> Board:
    board = Board()
    board.set_fen(_pad_fen(str(row.get("fen", ""))))
    board.set_search_context(
        int(row.get("search_plies", row.get("ply", 0)) or 0),
        int(row.get("no_capture_count", 0) or 0),
        int(row.get("repetition_count_hint", 1) or 1),
    )
    return board


def _move(row: dict[str, Any]) -> int | None:
    try:
        move = int(uci_move_to_internal(str(row.get("candidate_move", ""))[:4]))
    except Exception:
        return None
    return move


def _allows_guard(
    board: Board,
    move: int,
    guard: str,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    if guard == "mate1":
        return bool(_move_allows_opponent_mate1(
            board,
            int(move),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
    if guard == "mate2":
        return bool(_move_allows_opponent_check_forced_mate2(
            board,
            int(move),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
    if guard.startswith("forcing"):
        plies = int(guard.removeprefix("forcing"))
        return bool(_move_allows_opponent_forcing_check_win(
            board,
            int(move),
            plies_remaining=plies,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ))
    raise ValueError(f"unknown guard config: {guard}")


def _candidate_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected = _selected(rows)
    selected_move = str(selected.get("candidate_move", selected.get("selected_move", "")))[:4]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        move_s = str(row.get("candidate_move", ""))[:4]
        if not move_s or move_s == selected_move:
            continue
        if _rank(row.get("v13_visit_rank")) > int(top_k):
            continue
        if move_s in seen:
            continue
        seen.add(move_s)
        out.append(row)
    out.sort(key=lambda row: (_rank(row.get("v13_visit_rank")), -_float(row.get("v13_visit_prob"), 0.0)))
    return out


def _eval_one(
    key: str,
    rows: list[dict[str, Any]],
    guard: str,
    root_class: str,
    *,
    top_k: int,
    max_candidates: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> dict[str, Any]:
    selected = _selected(rows)
    board = _board_from_row(selected)
    sel_move_s = str(selected.get("candidate_move", selected.get("selected_move", "")))[:4]
    sel_move = _move(selected)
    sel_regret = _float(selected.get("regret_cp"), 0.0)
    root_bad = sel_regret >= 150.0
    root_cat = bool(root_bad and (sel_regret >= 1000.0 or selected.get("pika_mate_in_child") is not None))
    selected_flagged = False if sel_move is None else _allows_guard(
        board,
        sel_move,
        guard,
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    )

    replacement: dict[str, Any] | None = None
    checked = 0
    if selected_flagged:
        for cand in _candidate_rows(rows, top_k):
            if checked >= int(max_candidates):
                break
            cand_move = _move(cand)
            if cand_move is None or not bool(board.is_legal(int(cand_move))):
                continue
            checked += 1
            if not _allows_guard(
                board,
                cand_move,
                guard,
                max_plies=max_plies,
                repeat_limit=repeat_limit,
                repeat_min_ply=repeat_min_ply,
                no_capture_limit=no_capture_limit,
            ):
                replacement = cand
                break

    repl_regret = None if replacement is None else _float(replacement.get("regret_cp"), 0.0)
    prevented_safe = bool(root_bad and repl_regret is not None and repl_regret < 150.0)
    prevented_improved = bool(root_bad and repl_regret is not None and repl_regret <= max(sel_regret - 100.0, 0.0))
    regression = bool(
        (not root_bad)
        and repl_regret is not None
        and repl_regret >= 150.0
        and repl_regret >= sel_regret + 100.0
    )
    return {
        "root_key": key,
        "guard": guard,
        "root_class": root_class,
        "selected_move": sel_move_s,
        "teacher_best_move": str(selected.get("teacher_best_move", ""))[:4],
        "selected_regret_cp": sel_regret,
        "root_bad": bool(root_bad),
        "root_catastrophic": bool(root_cat),
        "selected_flagged": bool(selected_flagged),
        "checked_alternatives": int(checked),
        "replacement_move": None if replacement is None else str(replacement.get("candidate_move", ""))[:4],
        "replacement_regret_cp": repl_regret,
        "prevented_safe": prevented_safe,
        "prevented_improved": prevented_improved,
        "regression": regression,
        "source_jsonl": selected.get("_source_jsonl"),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    bad = [r for r in records if r["root_bad"]]
    clean = [r for r in records if not r["root_bad"]]
    cat = [r for r in records if r["root_catastrophic"]]
    horizon = [r for r in bad if r["root_class"] == "horizon_mate"]
    regressions = sum(1 for r in clean if r["regression"])
    return {
        "roots": len(records),
        "bad_roots": len(bad),
        "catastrophic_roots": len(cat),
        "horizon_mate_roots": len(horizon),
        "clean_roots": len(clean),
        "selected_flagged": sum(1 for r in records if r["selected_flagged"]),
        "bad_flagged": sum(1 for r in bad if r["selected_flagged"]),
        "catastrophic_flagged": sum(1 for r in cat if r["selected_flagged"]),
        "horizon_mate_flagged": sum(1 for r in horizon if r["selected_flagged"]),
        "bad_prevented_safe": sum(1 for r in bad if r["prevented_safe"]),
        "bad_prevented_improved": sum(1 for r in bad if r["prevented_improved"]),
        "catastrophic_prevented_safe": sum(1 for r in cat if r["prevented_safe"]),
        "horizon_mate_prevented_safe": sum(1 for r in horizon if r["prevented_safe"]),
        "clean_regressions": regressions,
        "clean_regression_rate_pct": 0.0 if not clean else 100.0 * regressions / len(clean),
        "class_counts": dict(Counter(r["root_class"] for r in records)),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V13.5 Root Mate/Horizon Veto Eval",
        "",
        "| guard | bad safe | bad improved | catastrophic safe | horizon safe | clean regressions | bad flagged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for guard, summary in payload["summaries"].items():
        lines.append(
            f"| {guard} | {summary['bad_prevented_safe']}/{summary['bad_roots']} | "
            f"{summary['bad_prevented_improved']}/{summary['bad_roots']} | "
            f"{summary['catastrophic_prevented_safe']}/{summary['catastrophic_roots']} | "
            f"{summary['horizon_mate_prevented_safe']}/{summary['horizon_mate_roots']} | "
            f"{summary['clean_regressions']}/{summary['clean_roots']} ({summary['clean_regression_rate_pct']:.2f}%) | "
            f"{summary['bad_flagged']}/{summary['bad_roots']} |"
        )
    lines += ["", "## Flagged Bad Roots", "", "| guard | regret | class | selected | replacement | safe |", "|---|---:|---|---|---|---|"]
    for rec in payload["records"]:
        if rec["root_bad"] and rec["selected_flagged"]:
            lines.append(
                f"| {rec['guard']} | {rec['selected_regret_cp']:.1f} | {rec['root_class']} | "
                f"{rec['selected_move']} | {rec['replacement_move']} | {rec['prevented_safe']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_regret_jsonl", nargs="+")
    parser.add_argument("--trajectory-json", default="")
    parser.add_argument("--guards", default="mate1,mate2,forcing5")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    args = parser.parse_args()

    groups = _load_groups([Path(p) for p in args.root_regret_jsonl])
    classes = _load_trajectory_classes(Path(args.trajectory_json) if args.trajectory_json else None)
    guards = [g.strip() for g in str(args.guards).split(",") if g.strip()]
    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for guard in guards:
        records = [
            _eval_one(
                key,
                rows,
                guard,
                classes.get(key, "unknown"),
                top_k=int(args.top_k),
                max_candidates=int(args.max_candidates),
                max_plies=int(args.max_plies),
                repeat_limit=int(args.repeat_limit),
                repeat_min_ply=int(args.repeat_min_ply),
                no_capture_limit=int(args.no_capture_limit),
            )
            for key, rows in sorted(groups.items())
        ]
        summaries[guard] = _summarize(records)
        all_records.extend(records)
    payload = {
        "root_regret_jsonl": [str(Path(p)) for p in args.root_regret_jsonl],
        "trajectory_json": str(args.trajectory_json or ""),
        "top_k": int(args.top_k),
        "max_candidates": int(args.max_candidates),
        "summaries": summaries,
        "records": all_records,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    printable = {k: v for k, v in payload.items() if k != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
