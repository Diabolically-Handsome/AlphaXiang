#!/usr/bin/env python3
"""Audit whether V13 losses still had early saveable root decisions.

This script consumes external_arena JSONs with shadow-value probes.  For every
recorded root decision in loss games, it reconstructs the board, builds a
candidate set from:

  - the actual selected move,
  - the recorded root best move,
  - WDL shadow top-K moves,
  - and any scalar/shadow verifier candidates logged online,

then scores candidate child positions with Pikafish.  The goal is not to prove
an alternate move would change the game outcome, but to locate the earliest
root where the selected move is already much worse than another available
candidate, and whether that signal appears early enough to be actionable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
_REPO = _TOOLS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool, PikafishPoolTimeout  # noqa: E402
from xiangqi_mcts_ext import Board  # noqa: E402

TERMINAL_ONGOING = -1


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


def _safe_move_uci(raw: Any) -> str:
    text = str(raw or "").strip()
    return text[:4] if len(text) >= 4 else ""


def _input_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("external_arena_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("no external_arena_*.json files found")
    return files


def _score(result: str) -> float:
    if result == "our_win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _board_before_ply(game: dict[str, Any], ply: int) -> Board:
    board = Board()
    board.set_fen(_pad_fen(str(game.get("opening_fen", ""))))
    for i, move_uci in enumerate(list(game.get("moves_uci", []) or [])[: max(0, int(ply))]):
        move = int(uci_move_to_internal(_safe_move_uci(move_uci)))
        if not bool(board.is_legal(move)):
            raise ValueError(
                f"illegal historical move while reconstructing game={game.get('index')} "
                f"ply={i} move={move_uci} fen={board.fen()}"
            )
        board.push(move)
    return board


def _terminal_child_eval_for_opponent(
    board: Board,
    move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> int | None:
    mover_is_red = int(board.turn()) == 0
    board.push(int(move))
    try:
        term = int(
            board.terminal_code(
                int(max_plies),
                int(repeat_limit),
                int(repeat_min_ply),
                int(no_capture_limit),
            )
        )
        if term == TERMINAL_ONGOING:
            return None
        red_result = int(board.terminal_result_red_view(term))
        if red_result == 0:
            return 0
        mover_won = (red_result > 0) == mover_is_red
        return -20000 if mover_won else 20000
    finally:
        board.pop()


@dataclass(frozen=True)
class EvalRef:
    record_index: int
    candidate_index: int


def _add_candidate(
    candidates: dict[str, dict[str, Any]],
    move_uci: str,
    *,
    source: str,
    rank: int | None = None,
    prob: float | None = None,
) -> None:
    move_uci = _safe_move_uci(move_uci)
    if not move_uci:
        return
    row = candidates.setdefault(
        move_uci,
        {
            "move_uci": move_uci,
            "sources": [],
            "ranks": {},
            "probs": {},
        },
    )
    if source not in row["sources"]:
        row["sources"].append(source)
    if rank is not None:
        row["ranks"][source] = int(rank)
    if prob is not None and math.isfinite(float(prob)):
        row["probs"][source] = float(prob)


def _add_logged_candidate(candidates: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    move_uci = _safe_move_uci(row.get("move_uci"))
    if not move_uci:
        return
    sources = list(row.get("sources", []) or ["logged_candidate"])
    ranks = row.get("ranks", {}) or {}
    probs = row.get("probs", {}) or {}
    for source in sources:
        _add_candidate(
            candidates,
            move_uci,
            source=str(source),
            rank=(int(ranks[source]) if source in ranks and str(ranks[source]).lstrip("-").isdigit() else None),
            prob=(float(probs[source]) if source in probs else None),
        )


def _candidate_child_fen(
    board: Board,
    move_uci: str,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[str | None, dict[str, Any] | None]:
    move = int(uci_move_to_internal(_safe_move_uci(move_uci)))
    if not bool(board.is_legal(move)):
        return None, {"move_uci": _safe_move_uci(move_uci), "illegal": True}
    terminal_cp = _terminal_child_eval_for_opponent(
        board,
        move,
        max_plies=int(max_plies),
        repeat_limit=int(repeat_limit),
        repeat_min_ply=int(repeat_min_ply),
        no_capture_limit=int(no_capture_limit),
    )
    if terminal_cp is not None:
        return None, {
            "move_uci": _safe_move_uci(move_uci),
            "child_eval_cp_opponent_pov": int(terminal_cp),
            "root_eval_cp_our_pov": int(-terminal_cp),
            "mate_in": None,
            "verifier_mode": "terminal",
        }
    board.push(move)
    try:
        return _pad_fen(board.fen()), None
    finally:
        board.pop()


def _collect_records(
    files: list[Path],
    *,
    results: set[str],
    side: str,
    top_k: int,
    max_records: int,
    max_records_per_game: int,
) -> tuple[list[dict[str, Any]], list[PikafishJob], dict[int, EvalRef]]:
    records: list[dict[str, Any]] = []
    jobs: list[PikafishJob] = []
    refs: dict[int, EvalRef] = {}
    next_job = 0
    per_game_counts: Counter[tuple[str, int]] = Counter()

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        max_plies = int(cfg.get("max_plies", 300))
        repeat_limit = int(cfg.get("repeat_limit", 6))
        repeat_min_ply = int(cfg.get("repeat_min_ply", 12))
        no_capture_limit = int(cfg.get("no_capture_limit", 120))

        for game in payload.get("per_game", []) or []:
            result = str(game.get("result", ""))
            if result not in results:
                continue
            if side != "any" and str(game.get("our_side", "")) != side:
                continue
            game_key = (str(path), int(game.get("index", -1)))
            for event in game.get("search_stats", []) or []:
                if int(max_records_per_game) > 0 and per_game_counts[game_key] >= int(max_records_per_game):
                    continue
                ply = int(event.get("ply", -1))
                if ply < 0:
                    continue
                probe = event.get("shadow_value_probe") if isinstance(event.get("shadow_value_probe"), dict) else {}
                attempt = (
                    probe.get("shadow_disagreement_verifier_attempt")
                    if isinstance(probe.get("shadow_disagreement_verifier_attempt"), dict)
                    else {}
                )
                actual_uci = _safe_move_uci(
                    probe.get("actual_move_uci")
                    or attempt.get("original_move_uci")
                    or event.get("move_uci")
                )
                if not actual_uci:
                    continue

                board = _board_before_ply(game, ply)
                candidates: dict[str, dict[str, Any]] = {}
                _add_candidate(candidates, actual_uci, source="actual", rank=1)
                _add_candidate(candidates, event.get("root_best_move_uci"), source="root_best", rank=1)
                _add_candidate(candidates, probe.get("shadow_best_move_uci"), source="shadow_best", rank=1)
                for row in list(probe.get("shadow_top_moves", []) or [])[: max(1, int(top_k))]:
                    _add_candidate(
                        candidates,
                        row.get("move_uci"),
                        source="shadow_topk",
                        rank=int(row.get("rank", 999)),
                        prob=float(row.get("visit_prob", 0.0) or 0.0),
                    )
                for row in list(attempt.get("candidates", []) or []):
                    _add_logged_candidate(candidates, row)

                record = {
                    "source_json": str(path),
                    "game_index": int(game.get("index", event.get("game_index", -1))),
                    "opening_id": str(game.get("opening_id", "")),
                    "opening_index": game.get("opening_index"),
                    "result": result,
                    "score": _score(result),
                    "termination": str(game.get("termination", "")),
                    "our_side": str(game.get("our_side", "")),
                    "ply": int(ply),
                    "selected_move_uci": actual_uci,
                    "root_best_move_uci": _safe_move_uci(event.get("root_best_move_uci")),
                    "root_value": event.get("root_value"),
                    "shadow_value_source": probe.get("shadow_value_source"),
                    "shadow_root_value": probe.get("shadow_root_value"),
                    "shadow_best_move_uci": _safe_move_uci(probe.get("shadow_best_move_uci")),
                    "shadow_disagrees": bool(probe.get("disagrees_with_actual")),
                    "actual_rank_in_shadow_topk": probe.get("actual_rank_in_shadow_topk"),
                    "actual_visit_prob_in_shadow_topk": probe.get("actual_visit_prob_in_shadow_topk"),
                    "online_attempted": bool(attempt.get("attempted")),
                    "online_accepted": bool(attempt.get("accepted")),
                    "online_reason": attempt.get("reason"),
                    "candidate_count": int(len(candidates)),
                    "candidates": list(candidates.values()),
                }
                record_index = len(records)
                for cand_i, cand in enumerate(record["candidates"]):
                    fen, terminal_row = _candidate_child_fen(
                        board,
                        str(cand["move_uci"]),
                        max_plies=max_plies,
                        repeat_limit=repeat_limit,
                        repeat_min_ply=repeat_min_ply,
                        no_capture_limit=no_capture_limit,
                    )
                    if terminal_row is not None:
                        cand.update(terminal_row)
                        continue
                    if fen is None:
                        cand["illegal"] = True
                        continue
                    job_index = next_job
                    next_job += 1
                    cand["_job_index"] = int(job_index)
                    refs[job_index] = EvalRef(record_index=record_index, candidate_index=cand_i)
                    jobs.append(PikafishJob(index=job_index, fen=fen, depth=0, multipv=1))
                records.append(record)
                per_game_counts[game_key] += 1
                if int(max_records) > 0 and len(records) >= int(max_records):
                    return records, jobs, refs
    return records, jobs, refs


def _fill_results(
    records: list[dict[str, Any]],
    jobs: list[PikafishJob],
    refs: dict[int, EvalRef],
    *,
    depth: int,
    nodes: int,
    movetime_ms: int,
    workers: int,
    threads_per_worker: int,
    hash_mb: int,
    binary: str,
    timeout_s: float,
) -> None:
    for job in jobs:
        job.depth = int(depth)
        job.nodes = int(nodes)
        job.movetime_ms = int(movetime_ms)
    if not jobs:
        return
    t0 = time.monotonic()
    with PikafishPool(
        num_workers=int(workers),
        binary_path=binary,
        threads_per_worker=int(threads_per_worker),
        hash_mb=int(hash_mb),
    ) as pool:
        pool.submit_all(jobs)
        try:
            results = pool.collect(
                len(jobs),
                timeout_s=float(timeout_s),
                progress_cb=lambda done, total: print(f"saveability pika: {done}/{total}", flush=True),
            )
        except PikafishPoolTimeout as exc:
            results = exc.partial_results
            print(f"WARNING: {exc}", flush=True)
    by_job = {int(row.index): row for row in results}
    mode = (
        f"depth{int(depth)}"
        if int(depth) > 0
        else f"nodes{int(nodes)}"
        if int(nodes) > 0
        else f"movetime{int(movetime_ms)}"
        if int(movetime_ms) > 0
        else "depth1"
    )
    for job_index, ref in refs.items():
        result = by_job.get(int(job_index))
        cand = records[int(ref.record_index)]["candidates"][int(ref.candidate_index)]
        if result is None:
            cand["error"] = "missing_pikafish_result"
        elif result.error:
            cand["error"] = result.error
        else:
            cand["child_eval_cp_opponent_pov"] = int(result.eval_cp)
            cand["root_eval_cp_our_pov"] = int(-result.eval_cp)
            cand["mate_in"] = result.mate_in
            cand["verifier_mode"] = mode
        cand.pop("_job_index", None)
    print(f"saveability pika completed {len(by_job)}/{len(jobs)} jobs in {time.monotonic() - t0:.1f}s", flush=True)


def _classify_records(
    records: list[dict[str, Any]],
    *,
    margin_cp: int,
    bad_cp: int,
    catastrophic_cp: int,
    saveable_max_cp: int,
    early_ply_max: int,
) -> dict[str, Any]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    all_improvements: list[int] = []

    for record in records:
        rows = [
            row
            for row in record.get("candidates", [])
            if row.get("child_eval_cp_opponent_pov") is not None and not row.get("illegal")
        ]
        selected = next((row for row in rows if str(row.get("move_uci")) == str(record.get("selected_move_uci"))), None)
        best = min(rows, key=lambda row: int(row["child_eval_cp_opponent_pov"]), default=None)
        if selected is None:
            status = "missing_selected_eval"
        elif best is None:
            status = "missing_best_eval"
        else:
            selected_cp = int(selected["child_eval_cp_opponent_pov"])
            best_cp = int(best["child_eval_cp_opponent_pov"])
            improvement = int(selected_cp - best_cp)
            record["selected_child_eval_cp_opponent_pov"] = selected_cp
            record["best_move_uci"] = str(best["move_uci"])
            record["best_child_eval_cp_opponent_pov"] = best_cp
            record["improvement_cp"] = improvement
            record["selected_mate_in_child"] = selected.get("mate_in")
            record["best_mate_in_child"] = best.get("mate_in")
            record["bad_selected"] = bool(selected_cp >= int(bad_cp) or selected.get("mate_in") is not None)
            record["catastrophic_selected"] = bool(selected_cp >= int(catastrophic_cp) or selected.get("mate_in") is not None)
            record["saveable_local"] = bool(improvement >= int(margin_cp))
            record["saveable_to_holdable"] = bool(improvement >= int(margin_cp) and best_cp <= int(saveable_max_cp))
            record["all_candidates_bad"] = bool(rows and min(int(row["child_eval_cp_opponent_pov"]) for row in rows) >= int(bad_cp))
            record["early"] = bool(int(record.get("ply", 999999)) <= int(early_ply_max))
            all_improvements.append(improvement)
            if record["saveable_to_holdable"]:
                status = "saveable_to_holdable"
            elif record["saveable_local"]:
                status = "saveable_local_only"
            elif record["all_candidates_bad"]:
                status = "all_candidates_bad"
            elif record["bad_selected"]:
                status = "bad_selected_no_clear_save"
            else:
                status = "no_clear_issue"
        record["saveability_status"] = status
        reason_counts[status] += 1
        by_game[int(record.get("game_index", -1))].append(record)

    games: list[dict[str, Any]] = []
    for game_index, rows in sorted(by_game.items()):
        rows_sorted = sorted(rows, key=lambda row: int(row.get("ply", 0)))
        first_bad = next((row for row in rows_sorted if row.get("bad_selected")), None)
        first_saveable = next((row for row in rows_sorted if row.get("saveable_to_holdable")), None)
        first_local = next((row for row in rows_sorted if row.get("saveable_local")), None)
        first_all_bad = next((row for row in rows_sorted if row.get("all_candidates_bad")), None)
        first_online_accept = next((row for row in rows_sorted if row.get("online_accepted")), None)
        last = rows_sorted[-1] if rows_sorted else {}
        games.append(
            {
                "game_index": game_index,
                "game_number": game_index + 1,
                "result": str(last.get("result", "")),
                "termination": str(last.get("termination", "")),
                "positions": len(rows_sorted),
                "first_bad_ply": None if first_bad is None else int(first_bad["ply"]),
                "first_saveable_to_holdable_ply": None if first_saveable is None else int(first_saveable["ply"]),
                "first_saveable_local_ply": None if first_local is None else int(first_local["ply"]),
                "first_all_candidates_bad_ply": None if first_all_bad is None else int(first_all_bad["ply"]),
                "first_online_accept_ply": None if first_online_accept is None else int(first_online_accept["ply"]),
                "early_saveable_to_holdable": bool(
                    first_saveable is not None and int(first_saveable["ply"]) <= int(early_ply_max)
                ),
                "first_bad_record": _brief_record(first_bad),
                "first_saveable_to_holdable_record": _brief_record(first_saveable),
                "first_saveable_local_record": _brief_record(first_local),
                "first_all_candidates_bad_record": _brief_record(first_all_bad),
            }
        )

    saveable_games = [g for g in games if g["first_saveable_to_holdable_ply"] is not None]
    early_saveable_games = [g for g in games if g["early_saveable_to_holdable"]]
    all_bad_games = [g for g in games if g["first_all_candidates_bad_ply"] is not None]
    return {
        "records": len(records),
        "games": len(games),
        "status_counts": dict(reason_counts),
        "saveable_games": len(saveable_games),
        "early_saveable_games": len(early_saveable_games),
        "all_bad_games": len(all_bad_games),
        "improvement_cp": {
            "mean": None if not all_improvements else mean(all_improvements),
            "median": None if not all_improvements else median(all_improvements),
            "min": None if not all_improvements else min(all_improvements),
            "max": None if not all_improvements else max(all_improvements),
        },
        "games_detail": games,
    }


def _brief_record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "ply": row.get("ply"),
        "selected": row.get("selected_move_uci"),
        "best": row.get("best_move_uci"),
        "selected_cp_opp": row.get("selected_child_eval_cp_opponent_pov"),
        "best_cp_opp": row.get("best_child_eval_cp_opponent_pov"),
        "improvement_cp": row.get("improvement_cp"),
        "root_value": row.get("root_value"),
        "shadow_root_value": row.get("shadow_root_value"),
        "shadow_disagrees": row.get("shadow_disagrees"),
        "online_reason": row.get("online_reason"),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# V13 Early Saveability Audit",
        "",
        f"- files: {payload['files']}",
        f"- records: {summary['records']}",
        f"- games: {summary['games']}",
        f"- verifier: {payload['config']['verifier_mode']}, workers={payload['config']['workers']} x threads={payload['config']['threads_per_worker']}",
        f"- margin_cp: {payload['config']['margin_cp']}",
        f"- bad_cp: {payload['config']['bad_cp']}",
        f"- saveable_max_cp: {payload['config']['saveable_max_cp']}",
        f"- early_ply_max: {payload['config']['early_ply_max']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| saveable games | {summary['saveable_games']} / {summary['games']} |",
        f"| early saveable games | {summary['early_saveable_games']} / {summary['games']} |",
        f"| games with all-candidate-bad point | {summary['all_bad_games']} / {summary['games']} |",
        f"| improvement cp mean/median/min/max | {summary['improvement_cp']} |",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in sorted(summary["status_counts"].items(), key=lambda item: (-int(item[1]), str(item[0]))):
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "## Per-Game First Signals",
        "",
        "| game | result | term | first bad | first saveable holdable | first local save | first all bad | online accept |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for game in summary["games_detail"]:
        lines.append(
            f"| {game['game_number']} | {game['result']} | {game['termination']} | "
            f"{game['first_bad_ply']} | {game['first_saveable_to_holdable_ply']} | "
            f"{game['first_saveable_local_ply']} | {game['first_all_candidates_bad_ply']} | "
            f"{game['first_online_accept_ply']} |"
        )
    lines += [
        "",
        "## Earliest Saveable-To-Holdable Records",
        "",
        "| game | ply | selected | best | selected cp opp | best cp opp | improvement | root V | shadow V |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for game in summary["games_detail"]:
        row = game.get("first_saveable_to_holdable_record")
        if not row:
            continue
        lines.append(
            f"| {game['game_number']} | {row.get('ply')} | {row.get('selected')} | {row.get('best')} | "
            f"{row.get('selected_cp_opp')} | {row.get('best_cp_opp')} | {row.get('improvement_cp')} | "
            f"{row.get('root_value')} | {row.get('shadow_root_value')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_or_dir", nargs="+")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--results", default="opp_win")
    parser.add_argument("--side", choices=["any", "red", "black"], default="black")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--margin-cp", type=int, default=300)
    parser.add_argument("--bad-cp", type=int, default=600)
    parser.add_argument("--catastrophic-cp", type=int, default=1000)
    parser.add_argument("--saveable-max-cp", type=int, default=300)
    parser.add_argument("--early-ply-max", type=int, default=80)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-records-per-game", type=int, default=0)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    args = parser.parse_args()

    files = _input_files(list(args.json_or_dir))
    results = {part.strip() for part in str(args.results).split(",") if part.strip()}
    records, jobs, refs = _collect_records(
        files,
        results=results,
        side=str(args.side),
        top_k=int(args.top_k),
        max_records=int(args.max_records),
        max_records_per_game=int(args.max_records_per_game),
    )
    print(f"collected records={len(records)} pikafish_jobs={len(jobs)} from files={len(files)}", flush=True)
    _fill_results(
        records,
        jobs,
        refs,
        depth=int(args.depth),
        nodes=int(args.nodes),
        movetime_ms=int(args.movetime_ms),
        workers=int(args.workers),
        threads_per_worker=int(args.threads_per_worker),
        hash_mb=int(args.hash_mb),
        binary=str(args.pikafish_binary),
        timeout_s=float(args.timeout_s),
    )
    summary = _classify_records(
        records,
        margin_cp=int(args.margin_cp),
        bad_cp=int(args.bad_cp),
        catastrophic_cp=int(args.catastrophic_cp),
        saveable_max_cp=int(args.saveable_max_cp),
        early_ply_max=int(args.early_ply_max),
    )
    verifier_mode = (
        f"depth{int(args.depth)}"
        if int(args.depth) > 0
        else f"nodes{int(args.nodes)}"
        if int(args.nodes) > 0
        else f"movetime{int(args.movetime_ms)}"
        if int(args.movetime_ms) > 0
        else "depth1"
    )
    payload = {
        "files": len(files),
        "config": {
            "results": sorted(results),
            "side": str(args.side),
            "top_k": int(args.top_k),
            "margin_cp": int(args.margin_cp),
            "bad_cp": int(args.bad_cp),
            "catastrophic_cp": int(args.catastrophic_cp),
            "saveable_max_cp": int(args.saveable_max_cp),
            "early_ply_max": int(args.early_ply_max),
            "verifier_mode": verifier_mode,
            "workers": int(args.workers),
            "threads_per_worker": int(args.threads_per_worker),
            "hash_mb": int(args.hash_mb),
            "pikafish_binary": str(args.pikafish_binary),
        },
        "summary": summary,
        "records": records,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
