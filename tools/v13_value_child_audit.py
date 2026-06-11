#!/usr/bin/env python3
"""Audit whether V13 policy top-K contains good moves but value ranks them badly.

For each extracted arena position where our model is to move:
  - run the checkpoint once at root to get legal policy top-K;
  - push each top-K candidate and score the child with the model value head;
  - ask Pikafish to evaluate the root and each child position;
  - compare policy rank and model-value rank against Pikafish child-Q ranking.

The key diagnostic split:
  - Pika best not in policy top-K -> policy/data issue.
  - Good candidate in policy top-K but model child-value ranks it low -> value issue.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import internal_move_to_uci, uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool, PikafishPoolTimeout, PikafishResult  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402

TERMINAL_ONGOING = -1


@dataclass
class Position:
    fen: str
    opening_fen: str
    moves_uci: list[str]
    ply: int
    game_index: int
    source_arena: str
    result: str
    termination: str
    our_side: str
    chosen_uci: str


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


def _reconstruct(opening_fen: str, moves: list[str], ply: int) -> Board | None:
    board = Board()
    if opening_fen:
        board.set_fen(_pad_fen(opening_fen))
    for uci in moves[:ply]:
        raw = int(uci_move_to_internal(str(uci)[:4]))
        if not bool(board.is_legal(raw)):
            return None
        board.push_legal(raw)
    return board


def _extract_positions(
    paths: list[Path],
    *,
    results: set[str],
    only_side: str,
    max_positions: int,
    ply_stride: int,
    min_ply: int,
    max_plies: int,
) -> list[Position]:
    out: list[Position] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload.get("per_game", []):
            if str(rec.get("result", "")) not in results:
                continue
            our_side = str(rec.get("our_side", ""))
            if only_side != "any" and our_side != only_side:
                continue
            opening_fen = str(rec.get("opening_fen") or "")
            moves = [str(move)[:4] for move in rec.get("moves_uci", [])]
            our_is_red = our_side == "red"
            board = Board()
            if opening_fen:
                board.set_fen(_pad_fen(opening_fen))
            our_turn_seen = 0
            for ply, uci in enumerate(moves[:max_plies]):
                red_to_move = int(board.turn()) == 0
                our_turn = red_to_move == our_is_red
                raw = int(uci_move_to_internal(uci))
                if our_turn:
                    if ply >= min_ply and (our_turn_seen % max(1, ply_stride) == 0):
                        out.append(
                            Position(
                                fen=_pad_fen(board.fen()),
                                opening_fen=_pad_fen(opening_fen) if opening_fen else "",
                                moves_uci=moves,
                                ply=int(ply),
                                game_index=int(rec.get("index", -1)),
                                source_arena=str(path),
                                result=str(rec.get("result", "")),
                                termination=str(rec.get("termination", "")),
                                our_side=our_side,
                                chosen_uci=uci,
                            )
                        )
                        if max_positions > 0 and len(out) >= max_positions:
                            return out
                    our_turn_seen += 1
                if not bool(board.is_legal(raw)):
                    break
                board.push_legal(raw)
    return out


def _load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def _forward(model, states: torch.Tensor, device: torch.device, batch_size: int, use_bfloat16: bool) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {"policy_logits": [], "value_scalar": []}
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for start in range(0, int(states.shape[0]), max(1, batch_size)):
        stop = min(start + max(1, batch_size), int(states.shape[0]))
        batch = states[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(batch)
        chunks["policy_logits"].append(out["policy_logits"].detach().cpu().float())
        chunks["value_scalar"].append(out["value_scalar"].detach().cpu().float())
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def _legal_policy_topk(logits: torch.Tensor, board: Board, top_k: int) -> tuple[list[int], list[float]]:
    legal = [int(m) for m in board.legal_moves()]
    stm_black = bool(int(board.turn()) == 1)
    idxs = torch.tensor([int(canonical_action(move, stm_black)) for move in legal], dtype=torch.long)
    legal_logits = logits[idxs]
    probs = torch.softmax(legal_logits.float(), dim=0)
    k = min(int(top_k), int(idxs.numel()))
    top = torch.topk(probs, k=k)
    out_moves = [int(legal[int(j.item())]) for j in top.indices]
    out_probs = [float(x) for x in top.values.tolist()]
    return out_moves, out_probs


def _terminal_q_cp(
    board: Board,
    root_stm_is_red: bool,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> float | None:
    term = int(board.terminal_code(int(max_plies), int(repeat_limit), int(repeat_min_ply), int(no_capture_limit)))
    if term == TERMINAL_ONGOING:
        return None
    red_result = int(board.terminal_result_red_view(term))
    if red_result == 0:
        return 0.0
    root_won = (red_result > 0) == bool(root_stm_is_red)
    return 20000.0 if root_won else -20000.0


def _rank_of(move_uci: str | None, rows: list[dict[str, Any]], key: str) -> int | None:
    if not move_uci:
        return None
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    for rank, row in enumerate(ordered, start=1):
        if str(row.get("move_uci")) == move_uci:
            return rank
    return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / (den_x * den_y)


def _collect_pika(pool: PikafishPool, jobs: list[PikafishJob], timeout_s: float) -> dict[int, PikafishResult]:
    pool.submit_all(jobs)
    try:
        results = pool.collect(len(jobs), timeout_s=float(timeout_s))
    except PikafishPoolTimeout as exc:
        results = exc.partial_results
    return {int(result.index): result for result in results}


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arena_json", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--results", default="opp_win")
    parser.add_argument("--only-side", choices=["any", "red", "black"], default="black")
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--ply-stride", type=int, default=4)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--pika-depth", type=int, default=10)
    parser.add_argument("--pika-root-multipv", type=int, default=8)
    parser.add_argument("--pika-workers", type=int, default=8)
    parser.add_argument("--pika-threads-per-worker", type=int, default=1)
    parser.add_argument("--pika-hash-mb", type=int, default=128)
    parser.add_argument("--pika-binary", default="/home/laure/pikafish/pikafish")
    parser.add_argument("--pika-timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-position-ms", type=float, default=0.0,
                        help="Reserved for summaries; does not interrupt individual positions.")
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    args = parser.parse_args()

    t0 = time.monotonic()
    paths = [Path(path) for path in args.arena_json]
    wanted = {part.strip() for part in str(args.results).split(",") if part.strip()}
    positions = _extract_positions(
        paths,
        results=wanted,
        only_side=str(args.only_side),
        max_positions=int(args.max_positions),
        ply_stride=int(args.ply_stride),
        min_ply=int(args.min_ply),
        max_plies=int(args.max_plies),
    )
    if not positions:
        raise SystemExit("no positions extracted")
    print(f"extracted {len(positions)} positions from {len(paths)} arena JSON(s)", flush=True)

    device = torch.device(args.device)
    print(f"loading checkpoint {args.checkpoint}", flush=True)
    model = _load_model(Path(args.checkpoint), device)
    root_states = []
    boards: list[Board] = []
    for pos in positions:
        board = Board()
        board.set_fen(pos.fen)
        boards.append(board)
        root_states.append(board.to_tensor_canonical().to(torch.float32)[0].contiguous())
    root_out = _forward(
        model,
        torch.stack(root_states, dim=0),
        device,
        batch_size=int(args.batch_size),
        use_bfloat16=not bool(args.disable_bf16),
    )

    candidate_rows_by_pos: list[list[dict[str, Any]]] = []
    child_states: list[torch.Tensor] = []
    child_state_refs: list[tuple[int, int, int]] = []
    root_jobs: list[PikafishJob] = []
    child_jobs: list[PikafishJob] = []
    child_job_refs: dict[int, tuple[int, int]] = {}
    job_index = 0

    for i, (pos, board) in enumerate(zip(positions, boards)):
        root_jobs.append(
            PikafishJob(
                index=job_index,
                fen=pos.fen,
                depth=int(args.pika_depth),
                multipv=int(args.pika_root_multipv),
            )
        )
        job_index += 1
        moves, probs = _legal_policy_topk(root_out["policy_logits"][i], board, int(args.top_k))
        chosen_raw = int(uci_move_to_internal(pos.chosen_uci))
        if chosen_raw not in moves and bool(board.is_legal(chosen_raw)):
            moves.append(chosen_raw)
            probs.append(0.0)
        rows: list[dict[str, Any]] = []
        root_stm_is_red = int(board.turn()) == 0
        for rank, (move, prob) in enumerate(zip(moves, probs), start=1):
            child = Board()
            child.set_fen(pos.fen)
            if not bool(child.is_legal(int(move))):
                continue
            child.push_legal(int(move))
            terminal_cp = _terminal_q_cp(
                child,
                root_stm_is_red,
                max_plies=int(args.max_plies),
                repeat_limit=int(args.repeat_limit),
                repeat_min_ply=int(args.repeat_min_ply),
                no_capture_limit=int(args.no_capture_limit),
            )
            row_idx = len(rows)
            row = {
                "move": int(move),
                "move_uci": internal_move_to_uci(int(move)),
                "policy_rank": int(rank),
                "policy_prob": float(prob),
                "is_chosen": internal_move_to_uci(int(move)) == pos.chosen_uci,
                "terminal_pika_q_cp": terminal_cp,
            }
            rows.append(row)
            child_states.append(child.to_tensor_canonical().to(torch.float32)[0].contiguous())
            child_state_refs.append((i, row_idx, int(move)))
            if terminal_cp is None:
                child_jobs.append(
                    PikafishJob(
                        index=job_index,
                        fen=_pad_fen(child.fen()),
                        depth=int(args.pika_depth),
                        multipv=1,
                    )
                )
                child_job_refs[job_index] = (i, row_idx)
                job_index += 1
        candidate_rows_by_pos.append(rows)

    print(f"model child evals: {len(child_states)} candidate child states", flush=True)
    child_out = _forward(
        model,
        torch.stack(child_states, dim=0),
        device,
        batch_size=int(args.batch_size),
        use_bfloat16=not bool(args.disable_bf16),
    )
    for child_i, (pos_i, row_i, _move) in enumerate(child_state_refs):
        child_v = float(child_out["value_scalar"][child_i].flatten()[0].item())
        candidate_rows_by_pos[pos_i][row_i]["model_child_value_opponent_pov"] = child_v
        candidate_rows_by_pos[pos_i][row_i]["model_q_root_pov"] = -child_v

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_jobs = root_jobs + child_jobs
    print(
        f"querying Pikafish depth={args.pika_depth}: roots={len(root_jobs)} children={len(child_jobs)} "
        f"workers={args.pika_workers}",
        flush=True,
    )
    pool = PikafishPool(
        num_workers=int(args.pika_workers),
        binary_path=str(args.pika_binary),
        threads_per_worker=int(args.pika_threads_per_worker),
        hash_mb=int(args.pika_hash_mb),
    )
    try:
        result_by_job = _collect_pika(pool, all_jobs, timeout_s=float(args.pika_timeout_s))
    finally:
        pool.close()

    root_job_ids = {job.index: i for i, job in enumerate(root_jobs)}
    root_pika: list[dict[str, Any] | None] = [None for _ in positions]
    for job_id, pos_i in root_job_ids.items():
        result = result_by_job.get(job_id)
        if result is None or result.error:
            continue
        root_pika[pos_i] = {
            "best_move_uci": str(result.best_move)[:4],
            "eval_cp": int(result.eval_cp),
            "mate_in": result.mate_in,
            "multipv": [
                {"move_uci": str(move)[:4], "score_cp": int(cp)}
                for move, cp in (result.multipv_moves or [])
            ],
        }
    for job_id, (pos_i, row_i) in child_job_refs.items():
        result = result_by_job.get(job_id)
        if result is None or result.error:
            candidate_rows_by_pos[pos_i][row_i]["pika_missing"] = True
            continue
        # Child eval is from opponent POV, so invert for root mover POV.
        candidate_rows_by_pos[pos_i][row_i]["pika_child_eval_opponent_pov_cp"] = int(result.eval_cp)
        candidate_rows_by_pos[pos_i][row_i]["pika_q_root_pov_cp"] = int(-result.eval_cp)
        candidate_rows_by_pos[pos_i][row_i]["pika_mate_in_child"] = result.mate_in
    for rows in candidate_rows_by_pos:
        for row in rows:
            if row.get("terminal_pika_q_cp") is not None:
                row["pika_child_eval_opponent_pov_cp"] = int(-float(row["terminal_pika_q_cp"]))
                row["pika_q_root_pov_cp"] = float(row["terminal_pika_q_cp"])
                row["pika_mate_in_child"] = None

    records: list[dict[str, Any]] = []
    n = 0
    policy_top1_hits = policy_top3_hits = policy_top5_hits = policy_topk_hits = 0
    value_top1_hits = value_top3_hits = 0
    value_bad_rank_when_policy_contains = 0
    policy_contains_but_value_miss = 0
    chosen_regrets: list[float] = []
    policy_regrets: list[float] = []
    value_regrets: list[float] = []
    pearsons: list[float] = []
    root_best_missing = 0

    for i, pos in enumerate(positions):
        rows = [row for row in candidate_rows_by_pos[i] if "pika_q_root_pov_cp" in row]
        if not rows:
            continue
        n += 1
        rows_by_pika = sorted(rows, key=lambda row: float(row["pika_q_root_pov_cp"]), reverse=True)
        rows_by_value = sorted(rows, key=lambda row: float(row["model_q_root_pov"]), reverse=True)
        rows_by_policy = sorted(rows, key=lambda row: int(row["policy_rank"]))
        teacher_best = rows_by_pika[0]
        policy_pick = rows_by_policy[0]
        value_pick = rows_by_value[0]
        best_cp = float(teacher_best["pika_q_root_pov_cp"])
        teacher_uci = str(teacher_best["move_uci"])
        root_best_uci = None if root_pika[i] is None else str(root_pika[i].get("best_move_uci") or "")[:4]
        root_best_rank = _rank_of(root_best_uci, rows, "policy_prob")
        teacher_policy_rank = int(teacher_best["policy_rank"])
        teacher_value_rank = _rank_of(teacher_uci, rows, "model_q_root_pov")

        if teacher_policy_rank <= 1:
            policy_top1_hits += 1
        if teacher_policy_rank <= 3:
            policy_top3_hits += 1
        if teacher_policy_rank <= 5:
            policy_top5_hits += 1
        if teacher_policy_rank <= int(args.top_k):
            policy_topk_hits += 1
        if teacher_value_rank is not None and teacher_value_rank <= 1:
            value_top1_hits += 1
        if teacher_value_rank is not None and teacher_value_rank <= 3:
            value_top3_hits += 1
        if root_best_uci and all(str(row["move_uci"]) != root_best_uci for row in rows):
            root_best_missing += 1
        if teacher_policy_rank <= int(args.top_k) and teacher_value_rank is not None and teacher_value_rank > 3:
            value_bad_rank_when_policy_contains += 1
        if teacher_policy_rank <= 5 and teacher_value_rank is not None and teacher_value_rank > 3:
            policy_contains_but_value_miss += 1

        policy_regrets.append(best_cp - float(policy_pick["pika_q_root_pov_cp"]))
        value_regrets.append(best_cp - float(value_pick["pika_q_root_pov_cp"]))
        chosen = next((row for row in rows if bool(row.get("is_chosen"))), None)
        if chosen is not None:
            chosen_regrets.append(best_cp - float(chosen["pika_q_root_pov_cp"]))
        corr = _pearson(
            [float(row["model_q_root_pov"]) for row in rows],
            [math.tanh(float(row["pika_q_root_pov_cp"]) / 500.0) for row in rows],
        )
        if corr is not None:
            pearsons.append(float(corr))

        records.append(
            {
                "position": {
                    "fen": pos.fen,
                    "ply": pos.ply,
                    "game_index": pos.game_index,
                    "source_arena": pos.source_arena,
                    "result": pos.result,
                    "termination": pos.termination,
                    "our_side": pos.our_side,
                    "chosen_uci": pos.chosen_uci,
                },
                "root_model_value": float(root_out["value_scalar"][i].flatten()[0].item()),
                "root_pika": root_pika[i],
                "root_pika_best_in_model_topk": bool(root_best_uci and any(str(row["move_uci"]) == root_best_uci for row in rows)),
                "root_pika_best_policy_rank": root_best_rank,
                "teacher_best_among_model_candidates": {
                    "move_uci": teacher_uci,
                    "policy_rank": teacher_policy_rank,
                    "value_rank": teacher_value_rank,
                    "pika_q_root_pov_cp": best_cp,
                    "model_q_root_pov": float(teacher_best["model_q_root_pov"]),
                },
                "policy_pick": {
                    "move_uci": str(policy_pick["move_uci"]),
                    "pika_regret_cp": best_cp - float(policy_pick["pika_q_root_pov_cp"]),
                    "pika_q_root_pov_cp": float(policy_pick["pika_q_root_pov_cp"]),
                },
                "value_pick": {
                    "move_uci": str(value_pick["move_uci"]),
                    "pika_regret_cp": best_cp - float(value_pick["pika_q_root_pov_cp"]),
                    "pika_q_root_pov_cp": float(value_pick["pika_q_root_pov_cp"]),
                },
                "candidate_rows": sorted(rows, key=lambda row: int(row["policy_rank"])),
            }
        )

    def avg(values: list[float]) -> float | None:
        return None if not values else float(mean(values))

    def med(values: list[float]) -> float | None:
        return None if not values else float(median(values))

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "arena_json": [str(path) for path in paths],
        "positions_extracted": len(positions),
        "positions_scored": n,
        "top_k": int(args.top_k),
        "pika_depth": int(args.pika_depth),
        "pika_root_multipv": int(args.pika_root_multipv),
        "policy_teacher_best_recall_within_model_candidates": {
            "top1": policy_top1_hits / n if n else None,
            "top3": policy_top3_hits / n if n else None,
            "top5": policy_top5_hits / n if n else None,
            f"top{int(args.top_k)}": policy_topk_hits / n if n else None,
        },
        "value_teacher_best_rank_within_model_candidates": {
            "top1": value_top1_hits / n if n else None,
            "top3": value_top3_hits / n if n else None,
            "bad_rank_gt3_when_policy_contains_topk": value_bad_rank_when_policy_contains / n if n else None,
            "policy_top5_but_value_rank_gt3": policy_contains_but_value_miss / n if n else None,
        },
        "root_pika_best_missing_from_model_topk_rate": root_best_missing / n if n else None,
        "regret_cp": {
            "policy_pick_mean": avg(policy_regrets),
            "policy_pick_median": med(policy_regrets),
            "value_pick_mean": avg(value_regrets),
            "value_pick_median": med(value_regrets),
            "chosen_mean": avg(chosen_regrets),
            "chosen_median": med(chosen_regrets),
        },
        "model_child_value_vs_pika_child_q": {
            "mean_position_pearson": avg(pearsons),
            "median_position_pearson": med(pearsons),
            "n_positions_with_corr": len(pearsons),
        },
        "elapsed_s": time.monotonic() - t0,
    }
    payload = {"summary": summary, "records": records}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# V13 Value Child Audit",
        "",
        f"- positions scored: {n}/{len(positions)}",
        f"- topK: {int(args.top_k)}",
        f"- Pikafish depth: {int(args.pika_depth)}",
        f"- elapsed: {summary['elapsed_s']:.1f}s",
        "",
        "## Policy Recall",
        "",
        f"- teacher-best among model candidates in policy top1: {_fmt_pct(summary['policy_teacher_best_recall_within_model_candidates']['top1'])}",
        f"- top3: {_fmt_pct(summary['policy_teacher_best_recall_within_model_candidates']['top3'])}",
        f"- top5: {_fmt_pct(summary['policy_teacher_best_recall_within_model_candidates']['top5'])}",
        f"- root Pika best missing from model topK: {_fmt_pct(summary['root_pika_best_missing_from_model_topk_rate'])}",
        "",
        "## Value Ranking",
        "",
        f"- teacher-best in value top1: {_fmt_pct(summary['value_teacher_best_rank_within_model_candidates']['top1'])}",
        f"- teacher-best in value top3: {_fmt_pct(summary['value_teacher_best_rank_within_model_candidates']['top3'])}",
        f"- policy top5 but value rank >3: {_fmt_pct(summary['value_teacher_best_rank_within_model_candidates']['policy_top5_but_value_rank_gt3'])}",
        f"- mean child-value/Pika-Q Pearson: {summary['model_child_value_vs_pika_child_q']['mean_position_pearson']}",
        "",
        "## Regret",
        "",
        f"- policy pick mean regret cp: {summary['regret_cp']['policy_pick_mean']}",
        f"- value pick mean regret cp: {summary['regret_cp']['value_pick_mean']}",
        f"- chosen move mean regret cp: {summary['regret_cp']['chosen_mean']}",
        "",
    ]
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_json}", flush=True)
    if args.out_md:
        print(f"wrote {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
