"""Scan arena losses for value drops around our moves.

Given one or more external_arena JSON files, this tool reconstructs every lost
game and asks Pikafish to evaluate positions immediately before and after each
of our moves.  Scores are normalized to our side-to-move perspective:

    cp_before_our = Pikafish eval before our move
    cp_after_our  = -Pikafish eval after our move, because opponent is to move
    drop_cp       = cp_before_our - cp_after_our

The JSON output is compatible with verified_failure_slice.py and
post_blunder_value_slice.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool, PikafishResult  # noqa: E402
from xiangqi_mcts_ext import Board  # noqa: E402


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


def _load_loss_games(paths: list[Path], *, depth_tag: str) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for arena_path in paths:
        payload = json.loads(arena_path.read_text(encoding="utf-8"))
        for rec in payload.get("per_game", []):
            if str(rec.get("result")) != "opp_win":
                continue
            idx = len(games)
            games.append({
                "gid": f"{arena_path.stem}:{rec.get('index', idx)}:{idx}",
                "index": idx,
                "source_arena_json": str(arena_path),
                "source_game_index": int(rec.get("index", idx)),
                "depth": depth_tag,
                "side": str(rec.get("our_side", "")),
                "opening_fen": str(rec.get("opening_fen") or ""),
                "moves": [str(move)[:4] for move in rec.get("moves_uci", [])],
                "plies": int(rec.get("plies", len(rec.get("moves_uci", [])))),
                "termination": str(rec.get("termination", "")),
            })
    return games


def _collect_eval_requests(games: list[dict[str, Any]], *, max_plies: int) -> tuple[list[dict[str, Any]], list[PikafishJob]]:
    rows: list[dict[str, Any]] = []
    jobs: list[PikafishJob] = []
    for game in games:
        board = Board()
        opening_fen = str(game.get("opening_fen") or "")
        if opening_fen:
            board.set_fen(_pad_fen(opening_fen))
        our_is_red = str(game.get("side")) == "red"
        moves = [str(move)[:4] for move in game.get("moves", [])]
        for ply, uci in enumerate(moves[:max_plies]):
            red_to_move = int(board.turn()) == 0
            our_turn = red_to_move == our_is_red
            try:
                raw = int(uci_move_to_internal(uci))
            except Exception:
                break
            if our_turn:
                if not bool(board.is_legal(raw)):
                    break
                before_fen = _pad_fen(board.fen())
                before_job_idx = len(jobs)
                jobs.append(PikafishJob(index=before_job_idx, fen=before_fen, depth=0))
                board.push_legal(raw)
                after_fen = _pad_fen(board.fen())
                after_job_idx = len(jobs)
                jobs.append(PikafishJob(index=after_job_idx, fen=after_fen, depth=0))
                rows.append({
                    "gid": str(game["gid"]),
                    "game_index": int(game["index"]),
                    "source_game_index": int(game.get("source_game_index", game["index"])),
                    "depth": str(game.get("depth", "")),
                    "our_side": str(game.get("side", "")),
                    "ply": int(ply),
                    "our_move": str(uci),
                    "fen_before": before_fen,
                    "fen_after": after_fen,
                    "before_job_idx": before_job_idx,
                    "after_job_idx": after_job_idx,
                })
            else:
                if not bool(board.is_legal(raw)):
                    break
                board.push_legal(raw)
    return rows, jobs


def _attach_results(rows: list[dict[str, Any]], results: list[PikafishResult]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_index = {int(result.index): result for result in results}
    out: list[dict[str, Any]] = []
    missing = 0
    errors = 0
    for row in rows:
        before = by_index.get(int(row["before_job_idx"]))
        after = by_index.get(int(row["after_job_idx"]))
        if before is None or after is None:
            missing += 1
            continue
        if before.error or after.error:
            errors += 1
            continue
        cp_before = int(before.eval_cp)
        cp_after_our = -int(after.eval_cp)
        item = dict(row)
        item.pop("before_job_idx", None)
        item.pop("after_job_idx", None)
        item.update({
            "bestmove_before": str(before.best_move)[:4],
            "cp_before_our": float(cp_before),
            "cp_after_our": float(cp_after_our),
            "drop_cp": float(cp_before - cp_after_our),
            "mate_before": before.mate_in,
            "mate_after_opponent_pov": after.mate_in,
        })
        out.append(item)
    return out, {"missing_rows": missing, "error_rows": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arena_json", nargs="+")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--depth-tag", default="d5")
    parser.add_argument("--eval-depth", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--max-wait-s", type=float, default=3600.0)
    args = parser.parse_args()

    t0 = time.monotonic()
    paths = [Path(path) for path in args.arena_json]
    games = _load_loss_games(paths, depth_tag=str(args.depth_tag))
    rows, jobs = _collect_eval_requests(games, max_plies=int(args.max_plies))
    for job in jobs:
        job.depth = int(args.eval_depth)

    print(
        f"loaded {len(games)} loss game(s), {len(rows)} our-turn rows, "
        f"{len(jobs)} Pikafish jobs @ depth={args.eval_depth}",
        flush=True,
    )
    if not jobs:
        raise SystemExit("no jobs to evaluate")

    pool = PikafishPool(
        num_workers=int(args.workers),
        binary_path=args.pikafish_binary,
        threads_per_worker=int(args.threads_per_worker),
        hash_mb=int(args.hash_mb),
    )
    try:
        pool.submit_all(jobs)
        raw_results = pool.collect(len(jobs), timeout_s=float(args.max_wait_s))
    finally:
        pool.close()

    eval_rows, row_stats = _attach_results(rows, raw_results)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "arena_loss_drop_scan",
        "arena_json": [str(path) for path in paths],
        "depth_tag": str(args.depth_tag),
        "eval_depth": int(args.eval_depth),
        "loss_games": games,
        "eval_rows": eval_rows,
        "stats": {
            "loss_games": len(games),
            "candidate_rows": len(rows),
            "eval_rows": len(eval_rows),
            "pikafish_jobs": len(jobs),
            "duration_s": time.monotonic() - t0,
            **row_stats,
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} rows={len(eval_rows)} dt={payload['stats']['duration_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
