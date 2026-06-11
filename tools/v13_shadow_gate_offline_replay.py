#!/usr/bin/env python3
"""Offline replay for V13.5 scalar/WDL shadow-gate candidates.

This tool reuses existing external_arena JSON logs.  For every root where the
recorded scalar move disagrees with the WDL shadow move, it reconstructs the
board, builds a small candidate set from the actual move plus WDL shadow top-K,
and batch-evaluates child positions with PikafishPool.

It is intentionally a replay of the information already present in older logs:
the exact online gate also had scalar root top-K, but those candidates were not
stored before the fixed instrumentation.  This replay still answers a high-value
question: when WDL disagrees, does a CPU-saturated Pikafish verifier find a clear
replacement among the actual move and WDL-suggested alternatives?
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

from pikafish_opponent import internal_move_to_uci, uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool, PikafishPoolTimeout  # noqa: E402
from xiangqi_mcts_ext import Board  # noqa: E402

TERMINAL_ONGOING = -1


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


def _terminal_child_eval_for_opponent(
    board: Board,
    move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> int | None:
    mover_is_red = (int(board.turn()) == 0)
    board.push(int(move))
    try:
        term = int(board.terminal_code(
            int(max_plies),
            int(repeat_limit),
            int(repeat_min_ply),
            int(no_capture_limit),
        ))
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
    attempt_index: int
    move_uci: str


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


def _safe_move_uci(raw: Any) -> str:
    text = str(raw or "").strip()
    return text[:4] if len(text) >= 4 else ""


def _add_candidate(candidates: dict[str, dict[str, Any]], move_uci: str, *, source: str, rank: int | None = None, prob: float | None = None) -> None:
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


def _board_before_ply(game: dict[str, Any], ply: int) -> Board:
    board = Board()
    board.set_fen(_pad_fen(str(game.get("opening_fen", ""))))
    moves = list(game.get("moves_uci", []) or [])
    for i, move_uci in enumerate(moves[: max(0, int(ply))]):
        move = int(uci_move_to_internal(_safe_move_uci(move_uci)))
        if not bool(board.is_legal(move)):
            raise ValueError(
                f"illegal historical move while reconstructing game={game.get('index')} "
                f"ply={i} move={move_uci} fen={board.fen()}"
            )
        board.push(move)
    return board


def _candidate_child_fen(board: Board, move_uci: str, *, max_plies: int, repeat_limit: int, repeat_min_ply: int, no_capture_limit: int) -> tuple[str | None, dict[str, Any] | None]:
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


def _collect_attempts(files: list[Path], *, top_k: int, max_probes: int, only_losses: bool) -> tuple[list[dict[str, Any]], list[PikafishJob], dict[int, EvalRef]]:
    attempts: list[dict[str, Any]] = []
    jobs: list[PikafishJob] = []
    refs: dict[int, EvalRef] = {}
    next_job = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        max_plies = int(cfg.get("max_plies", 300))
        repeat_limit = int(cfg.get("repeat_limit", 6))
        repeat_min_ply = int(cfg.get("repeat_min_ply", 12))
        no_capture_limit = int(cfg.get("no_capture_limit", 120))
        for game in payload.get("per_game", []) or []:
            result = str(game.get("result", ""))
            if only_losses and result != "opp_win":
                continue
            for event in game.get("search_stats", []) or []:
                probe = event.get("shadow_value_probe")
                if not isinstance(probe, dict) or not bool(probe.get("disagrees_with_actual")):
                    continue
                ply = int(event.get("ply", -1))
                if ply < 0:
                    continue
                board = _board_before_ply(game, ply)
                actual_uci = _safe_move_uci(probe.get("actual_move_uci") or event.get("move_uci"))
                candidates: dict[str, dict[str, Any]] = {}
                _add_candidate(candidates, actual_uci, source="actual", rank=1)
                for row in list(probe.get("shadow_top_moves", []) or [])[: max(1, int(top_k))]:
                    _add_candidate(
                        candidates,
                        str(row.get("move_uci", "")),
                        source="shadow_topk",
                        rank=int(row.get("rank", 999)),
                        prob=float(row.get("visit_prob", 0.0) or 0.0),
                    )

                attempt = {
                    "source_json": str(path),
                    "game_index": int(game.get("index", event.get("game_index", -1))),
                    "ply": int(ply),
                    "side": str(event.get("side", "")),
                    "result": result,
                    "score": _score(result),
                    "termination": str(game.get("termination", "")),
                    "opening_id": str(game.get("opening_id", "")),
                    "opening_index": game.get("opening_index"),
                    "actual_move_uci": actual_uci,
                    "shadow_best_move_uci": _safe_move_uci(probe.get("shadow_best_move_uci")),
                    "shadow_root_value": probe.get("shadow_root_value"),
                    "root_value": event.get("root_value"),
                    "candidate_count": int(len(candidates)),
                    "candidates": list(candidates.values()),
                }
                attempt_index = len(attempts)
                for cand in attempt["candidates"]:
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
                    refs[job_index] = EvalRef(attempt_index=attempt_index, move_uci=str(cand["move_uci"]))
                    jobs.append(PikafishJob(index=job_index, fen=fen, depth=0, multipv=1))
                attempts.append(attempt)
                if int(max_probes) > 0 and len(attempts) >= int(max_probes):
                    return attempts, jobs, refs
    return attempts, jobs, refs


def _fill_results(attempts: list[dict[str, Any]], jobs: list[PikafishJob], refs: dict[int, EvalRef], *, depth: int, nodes: int, movetime_ms: int, workers: int, threads_per_worker: int, hash_mb: int, binary: str, timeout_s: float) -> None:
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
                progress_cb=lambda done, total: print(f"pikafish replay: {done}/{total}", flush=True),
            )
        except PikafishPoolTimeout as exc:
            results = exc.partial_results
            print(f"WARNING: {exc}", flush=True)
    by_job = {int(row.index): row for row in results}
    mode = f"depth{int(depth)}" if int(depth) > 0 else f"nodes{int(nodes)}" if int(nodes) > 0 else f"movetime{int(movetime_ms)}" if int(movetime_ms) > 0 else "depth1"
    for job_index, ref in refs.items():
        result = by_job.get(int(job_index))
        attempt = attempts[int(ref.attempt_index)]
        for cand in attempt["candidates"]:
            if str(cand.get("move_uci")) != str(ref.move_uci):
                continue
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
            break
    print(f"pikafish replay completed {len(by_job)}/{len(jobs)} jobs in {time.monotonic() - t0:.1f}s", flush=True)


def _classify_attempts(
    attempts: list[dict[str, Any]],
    *,
    margin_cp: int,
    mate_risk_margin_cp: int,
    mate_risk_cp: int,
    escape_margin_cp: int,
    escape_risk_cp: int,
    escape_safe_cp: int,
) -> dict[str, Any]:
    improvements: list[int] = []
    accepted = 0
    reason_counts: Counter[str] = Counter()
    by_result: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_rules: Counter[str] = Counter()
    for attempt in attempts:
        actual_uci = str(attempt.get("actual_move_uci", ""))
        rows = [
            row for row in attempt.get("candidates", [])
            if row.get("child_eval_cp_opponent_pov") is not None and not row.get("illegal")
        ]
        original = next((row for row in rows if str(row.get("move_uci")) == actual_uci), None)
        best = min(rows, key=lambda row: int(row["child_eval_cp_opponent_pov"]), default=None)
        if original is None:
            reason = "missing_original_eval"
            improvement = None
        elif best is None:
            reason = "missing_best_eval"
            improvement = None
        else:
            improvement = int(original["child_eval_cp_opponent_pov"]) - int(best["child_eval_cp_opponent_pov"])
            original_cp = int(original["child_eval_cp_opponent_pov"])
            best_cp = int(best["child_eval_cp_opponent_pov"])
            original_mate_risk = (
                original.get("mate_in") is not None
                or original_cp >= int(mate_risk_cp)
            )
            replacement_mate_risk = (
                best.get("mate_in") is not None
                or best_cp >= int(mate_risk_cp)
            )
            ordinary_accept = int(improvement) >= int(margin_cp)
            mate_risk_accept = (
                int(mate_risk_margin_cp) >= 0
                and bool(original_mate_risk)
                and not bool(replacement_mate_risk)
                and int(improvement) >= int(mate_risk_margin_cp)
            )
            escape_accept = (
                int(escape_margin_cp) >= 0
                and original_cp >= int(escape_risk_cp)
                and best_cp <= int(escape_safe_cp)
                and int(improvement) >= int(escape_margin_cp)
            )
            attempt["original_child_eval_cp_opponent_pov"] = int(original["child_eval_cp_opponent_pov"])
            attempt["best_move_uci"] = str(best["move_uci"])
            attempt["best_child_eval_cp_opponent_pov"] = int(best["child_eval_cp_opponent_pov"])
            attempt["improvement_cp"] = int(improvement)
            attempt["original_mate_risk"] = bool(original_mate_risk)
            attempt["replacement_mate_risk"] = bool(replacement_mate_risk)
            attempt["ordinary_accept"] = bool(ordinary_accept)
            attempt["mate_risk_accept"] = bool(mate_risk_accept)
            attempt["escape_accept"] = bool(escape_accept)
            if str(best["move_uci"]) == actual_uci:
                reason = "verified_original_best"
            elif bool(ordinary_accept or mate_risk_accept or escape_accept):
                reason = "accepted"
                if bool(mate_risk_accept and not ordinary_accept):
                    attempt["acceptance_rule"] = "mate_risk"
                elif bool(escape_accept and not ordinary_accept):
                    attempt["acceptance_rule"] = "escape"
                else:
                    attempt["acceptance_rule"] = "ordinary"
                accepted_rules[str(attempt["acceptance_rule"])] += 1
                accepted += 1
                improvements.append(int(improvement))
            else:
                reason = "improvement_below_margin"
        attempt["accepted"] = reason == "accepted"
        attempt["reason"] = reason
        reason_counts[reason] += 1
        by_result[str(attempt.get("result", ""))][reason] += 1
    all_improvements = [int(a["improvement_cp"]) for a in attempts if a.get("improvement_cp") is not None]
    return {
        "attempts": len(attempts),
        "accepted": accepted,
        "accepted_rate": 0.0 if not attempts else accepted / len(attempts),
        "reason_counts": dict(reason_counts),
        "reason_counts_by_result": {k: dict(v) for k, v in by_result.items()},
        "accepted_rules": dict(accepted_rules),
        "improvement_cp": {
            "mean": None if not all_improvements else mean(all_improvements),
            "median": None if not all_improvements else median(all_improvements),
            "min": None if not all_improvements else min(all_improvements),
            "max": None if not all_improvements else max(all_improvements),
        },
        "accepted_improvement_cp": {
            "mean": None if not improvements else mean(improvements),
            "median": None if not improvements else median(improvements),
            "min": None if not improvements else min(improvements),
            "max": None if not improvements else max(improvements),
        },
        "accepted_by_result": dict(Counter(str(a.get("result", "")) for a in attempts if a.get("accepted"))),
        "accepted_by_termination": dict(Counter(str(a.get("termination", "")) for a in attempts if a.get("accepted"))),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# V13.5 Shadow Gate Offline Replay",
        "",
        f"- files: {payload['files']}",
        f"- attempts: {s['attempts']}",
        f"- accepted: {s['accepted']} ({100.0 * s['accepted_rate']:.1f}%)",
        f"- margin_cp: {payload['config']['margin_cp']}",
        f"- mate_risk_margin_cp: {payload['config']['mate_risk_margin_cp']}",
        f"- mate_risk_cp: {payload['config']['mate_risk_cp']}",
        f"- escape_margin_cp: {payload['config']['escape_margin_cp']}",
        f"- escape_risk_cp: {payload['config']['escape_risk_cp']}",
        f"- escape_safe_cp: {payload['config']['escape_safe_cp']}",
        f"- verifier: {payload['config']['verifier_mode']}, workers={payload['config']['workers']} x threads={payload['config']['threads_per_worker']}",
        "",
        "## Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in sorted(s["reason_counts"].items(), key=lambda item: (-int(item[1]), str(item[0]))):
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Improvement CP",
        "",
        f"- all mean/median/min/max: {s['improvement_cp']}",
        f"- accepted mean/median/min/max: {s['accepted_improvement_cp']}",
        f"- accepted rules: {s['accepted_rules']}",
        "",
        "## Top Accepted Replacements",
        "",
        "| source | game | ply | result | term | actual | replacement | rule | improvement cp | original cp | replacement cp | opening |",
        "|---|---:|---:|---|---|---|---|---|---:|---:|---:|---|",
    ]
    accepted = sorted(
        [a for a in payload["attempts"] if a.get("accepted")],
        key=lambda a: -int(a.get("improvement_cp", 0)),
    )
    for a in accepted[:40]:
        source = Path(str(a.get("source_json", ""))).name
        lines.append(
            f"| {source} | {a.get('game_index')} | {a.get('ply')} | {a.get('result')} | "
            f"{a.get('termination')} | {a.get('actual_move_uci')} | {a.get('best_move_uci')} | "
            f"{a.get('acceptance_rule', '')} | {a.get('improvement_cp')} | {a.get('original_child_eval_cp_opponent_pov')} | "
            f"{a.get('best_child_eval_cp_opponent_pov')} | {a.get('opening_id')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_or_dir", nargs="+")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--margin-cp", type=int, default=100)
    parser.add_argument("--mate-risk-margin-cp", type=int, default=-1)
    parser.add_argument("--mate-risk-cp", type=int, default=19000)
    parser.add_argument("--escape-margin-cp", type=int, default=-1)
    parser.add_argument("--escape-risk-cp", type=int, default=500)
    parser.add_argument("--escape-safe-cp", type=int, default=100)
    parser.add_argument("--max-probes", type=int, default=0)
    parser.add_argument("--only-losses", action="store_true")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    args = parser.parse_args()

    files = _input_files(list(args.json_or_dir))
    attempts, jobs, refs = _collect_attempts(
        files,
        top_k=int(args.top_k),
        max_probes=int(args.max_probes),
        only_losses=bool(args.only_losses),
    )
    print(f"collected attempts={len(attempts)} pikafish_jobs={len(jobs)} from files={len(files)}", flush=True)
    _fill_results(
        attempts,
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
    summary = _classify_attempts(
        attempts,
        margin_cp=int(args.margin_cp),
        mate_risk_margin_cp=int(args.mate_risk_margin_cp),
        mate_risk_cp=int(args.mate_risk_cp),
        escape_margin_cp=int(args.escape_margin_cp),
        escape_risk_cp=int(args.escape_risk_cp),
        escape_safe_cp=int(args.escape_safe_cp),
    )
    verifier_mode = (
        f"depth{int(args.depth)}" if int(args.depth) > 0
        else f"nodes{int(args.nodes)}" if int(args.nodes) > 0
        else f"movetime{int(args.movetime_ms)}" if int(args.movetime_ms) > 0
        else "depth1"
    )
    payload = {
        "files": len(files),
        "config": {
            "top_k": int(args.top_k),
            "margin_cp": int(args.margin_cp),
            "mate_risk_margin_cp": int(args.mate_risk_margin_cp),
            "mate_risk_cp": int(args.mate_risk_cp),
            "escape_margin_cp": int(args.escape_margin_cp),
            "escape_risk_cp": int(args.escape_risk_cp),
            "escape_safe_cp": int(args.escape_safe_cp),
            "max_probes": int(args.max_probes),
            "only_losses": bool(args.only_losses),
            "verifier_mode": verifier_mode,
            "workers": int(args.workers),
            "threads_per_worker": int(args.threads_per_worker),
            "hash_mb": int(args.hash_mb),
            "pikafish_binary": str(args.pikafish_binary),
        },
        "summary": summary,
        "attempts": attempts,
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
