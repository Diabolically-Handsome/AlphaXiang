#!/usr/bin/env python3
"""Root-decision candidate-discovery audit for V13 arena losses.

This is deliberately different from the narrower value-child audit.  The core
question here is whether the move that a stronger oracle likes was even visible
to V13's root decision process.

For each extracted arena root position:
  - rerun V13 MCTS and keep root visit candidates;
  - collect raw model policy top-K;
  - collect Pikafish root MultiPV;
  - add all legal checks, captures, and immediate terminal wins;
  - evaluate every union candidate's child position with Pikafish;
  - classify failures as missing-candidate, ranking, or verifier-candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
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
from xiangqi_mcts_ext import (  # noqa: E402
    Board,
    canonical_action,
    make_gpu_evaluator,
    mcts_search,
    mcts_search_with_root_stats,
)
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


TERMINAL_ONGOING = -1
TERMINAL_CP = 20000.0


@dataclass
class OpeningContext:
    fen: str
    plies: int = 0
    no_capture_count: int = 0
    repetition_count_hint: int = 1


@dataclass
class Position:
    fen: str
    ply: int
    search_plies: int
    no_capture_count: int
    repetition_count_hint: int
    game_index: int
    source_arena: str
    result: str
    termination: str
    our_side: str
    chosen_uci: str
    opening_id: str = ""
    opening_index: int | None = None
    opening_fen: str = ""
    moves_uci: list[str] = field(default_factory=list)
    logged_search: dict[str, Any] | None = None


@dataclass
class Candidate:
    move: int
    move_uci: str
    sources: set[str] = field(default_factory=set)
    source_details: dict[str, Any] = field(default_factory=dict)


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


def _resolve_path(path: str | Path, base: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    first = (base / raw).resolve()
    if first.exists():
        return first
    return (_REPO / raw).resolve()


def _load_opening_contexts(path: Path, *, max_openings: int = 0) -> tuple[dict[int, OpeningContext], dict[str, OpeningContext]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("positions", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        return {}, {}

    by_index: dict[int, OpeningContext] = {}
    by_id: dict[str, OpeningContext] = {}
    for idx, raw in enumerate(raw_entries):
        if int(max_openings) > 0 and idx >= int(max_openings):
            break
        if not isinstance(raw, dict):
            continue
        fen = str(raw.get("fen", "")).strip()
        if not fen:
            continue
        ctx = OpeningContext(
            fen=_pad_fen(fen),
            plies=int(raw.get("plies", 0)),
            no_capture_count=int(raw.get("no_capture_count", 0)),
            repetition_count_hint=int(raw.get("repetition_count_hint", 1)),
        )
        by_index[idx] = ctx
        by_id[str(raw.get("id", f"opening_{idx + 1:03d}"))] = ctx
    return by_index, by_id


def _opening_context_for_record(
    record: dict[str, Any],
    *,
    suite_by_index: dict[int, OpeningContext],
    suite_by_id: dict[str, OpeningContext],
) -> OpeningContext:
    opening_id = str(record.get("opening_id", "") or "")
    opening_index = record.get("opening_index")
    if opening_id and opening_id in suite_by_id:
        return suite_by_id[opening_id]
    if opening_index is not None:
        try:
            idx = int(opening_index)
            if idx in suite_by_index:
                return suite_by_index[idx]
        except (TypeError, ValueError):
            pass
    return OpeningContext(fen=_pad_fen(str(record.get("opening_fen", "") or Board().fen())))


def _board_from_context(ctx: OpeningContext) -> Board:
    board = Board()
    board.set_fen(_pad_fen(ctx.fen))
    board.set_search_context(
        int(ctx.plies),
        int(ctx.no_capture_count),
        int(ctx.repetition_count_hint),
    )
    return board


def _board_from_position(pos: Position) -> Board:
    board = Board()
    board.set_fen(_pad_fen(pos.fen))
    board.set_search_context(
        int(pos.search_plies),
        int(pos.no_capture_count),
        int(pos.repetition_count_hint),
    )
    return board


def _extract_positions(
    paths: list[Path],
    *,
    results: set[str],
    only_side: str,
    max_positions: int,
    max_positions_per_file: int,
    ply_stride: int,
    min_ply: int,
    max_plies: int,
    opening_suite_path: str,
) -> list[Position]:
    out: list[Position] = []
    for path in paths:
        file_out = 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        suite_path_s = str(opening_suite_path or config.get("opening_suite_path", "") or "")
        suite_by_index: dict[int, OpeningContext] = {}
        suite_by_id: dict[str, OpeningContext] = {}
        if suite_path_s:
            suite_path = _resolve_path(suite_path_s, path.parent)
            if suite_path.exists():
                suite_by_index, suite_by_id = _load_opening_contexts(
                    suite_path,
                    max_openings=int(config.get("max_openings", 0) or 0),
                )

        for rec in payload.get("per_game", []) or []:
            if int(max_positions_per_file) > 0 and file_out >= int(max_positions_per_file):
                break
            if str(rec.get("result", "")) not in results:
                continue
            our_side = str(rec.get("our_side", ""))
            if only_side != "any" and our_side != only_side:
                continue
            moves = [str(move)[:4] for move in rec.get("moves_uci", [])]
            logged_by_ply = {
                int(row.get("ply", -1)): dict(row)
                for row in rec.get("search_stats", []) or []
                if isinstance(row, dict) and row.get("root_stats")
            }
            our_is_red = our_side == "red"
            ctx = _opening_context_for_record(rec, suite_by_index=suite_by_index, suite_by_id=suite_by_id)
            board = _board_from_context(ctx)
            our_turn_seen = 0
            for ply, uci in enumerate(moves[:max_plies]):
                red_to_move = int(board.turn()) == 0
                our_turn = red_to_move == our_is_red
                raw = int(uci_move_to_internal(uci))
                if our_turn:
                    if ply >= min_ply and (our_turn_seen % max(1, int(ply_stride)) == 0):
                        out.append(
                            Position(
                                fen=_pad_fen(board.fen()),
                                ply=int(ply),
                                search_plies=int(board.plies_played()),
                                no_capture_count=int(board.no_capture_count()),
                                repetition_count_hint=int(board.current_repetition_count()),
                                game_index=int(rec.get("index", -1)),
                                source_arena=str(path),
                                result=str(rec.get("result", "")),
                                termination=str(rec.get("termination", "")),
                                our_side=our_side,
                                chosen_uci=uci,
                                opening_id=str(rec.get("opening_id", "") or ""),
                                opening_index=None if rec.get("opening_index") is None else int(rec.get("opening_index")),
                                opening_fen=_pad_fen(ctx.fen),
                                moves_uci=moves,
                                logged_search=logged_by_ply.get(int(ply)),
                            )
                        )
                        file_out += 1
                        if int(max_positions) > 0 and len(out) >= int(max_positions):
                            return out
                        if int(max_positions_per_file) > 0 and file_out >= int(max_positions_per_file):
                            break
                    our_turn_seen += 1
                if not bool(board.is_legal(raw)):
                    break
                board.push_legal(raw)
    return out


def _extract_positions_from_audits(
    paths: list[Path],
    *,
    results: set[str],
    only_side: str,
    max_positions: int,
    max_positions_per_file: int,
) -> list[Position]:
    out: list[Position] = []
    for path in paths:
        file_out = 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload.get("records", []) or []:
            if int(max_positions_per_file) > 0 and file_out >= int(max_positions_per_file):
                break
            pos = rec.get("position", {})
            if str(pos.get("result", "")) not in results:
                continue
            our_side = str(pos.get("our_side", ""))
            if only_side != "any" and our_side != only_side:
                continue
            out.append(
                Position(
                    fen=_pad_fen(str(pos.get("fen", ""))),
                    ply=int(pos.get("ply", 0)),
                    search_plies=int(pos.get("search_plies", pos.get("ply", 0) or 0)),
                    no_capture_count=int(pos.get("no_capture_count", 0) or 0),
                    repetition_count_hint=int(pos.get("repetition_count_hint", 1) or 1),
                    game_index=int(pos.get("game_index", -1)),
                    source_arena=str(pos.get("source_arena", path)),
                    result=str(pos.get("result", "")),
                    termination=str(pos.get("termination", "")),
                    our_side=our_side,
                    chosen_uci=str(pos.get("chosen_uci", ""))[:4],
                    opening_id=str(pos.get("opening_id", "") or ""),
                    opening_index=None if pos.get("opening_index") is None else int(pos.get("opening_index")),
                    opening_fen=_pad_fen(str(pos.get("opening_fen", "") or Board().fen())),
                    moves_uci=[],
                )
            )
            file_out += 1
            if int(max_positions) > 0 and len(out) >= int(max_positions):
                return out
    return out


def _load_model_and_evaluator(checkpoint: Path, device: torch.device, use_bfloat16: bool):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    evaluator = make_gpu_evaluator(model, device=str(device), use_bfloat16=bool(use_bfloat16))
    return model, evaluator, int(state.get("global_step", 0))


@torch.inference_mode()
def _forward(model, states: torch.Tensor, device: torch.device, batch_size: int, use_bfloat16: bool) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {"policy_logits": [], "value_scalar": []}
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for start in range(0, int(states.shape[0]), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), int(states.shape[0]))
        batch = states[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(batch)
        chunks["policy_logits"].append(out["policy_logits"].detach().cpu().float())
        chunks["value_scalar"].append(out["value_scalar"].detach().cpu().float())
    return {key: torch.cat(parts, dim=0) for key, parts in chunks.items()}


def _legal_policy_topk(logits: torch.Tensor, board: Board, top_k: int) -> list[tuple[int, float]]:
    legal = [int(m) for m in board.legal_moves()]
    if not legal:
        return []
    stm_black = bool(int(board.turn()) == 1)
    idxs = torch.tensor([int(canonical_action(move, stm_black)) for move in legal], dtype=torch.long)
    legal_logits = logits[idxs].float()
    probs = torch.softmax(legal_logits, dim=0)
    k = min(int(top_k), int(idxs.numel()))
    top = torch.topk(probs, k=k)
    return [(int(legal[int(j.item())]), float(prob)) for prob, j in zip(top.values.tolist(), top.indices)]


def _root_policy_candidates_from_search(idxs, probs, board: Board) -> list[tuple[int, float]]:
    stm_black = bool(int(board.turn()) == 1)
    out: list[tuple[int, float]] = []
    for mv, prob in zip(list(idxs), list(probs)):
        raw_mv = int(canonical_action(int(mv), stm_black))
        out.append((raw_mv, float(prob)))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def _root_stats_candidates(root_stats: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for raw in root_stats:
        row = dict(raw)
        move = int(row.get("move_raw", -1))
        if move < 0:
            continue
        rows.append((move, row))
    rows.sort(key=lambda item: float(item[1].get("visit_prob", 0.0)), reverse=True)
    return rows


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
    return TERMINAL_CP if root_won else -TERMINAL_CP


def _move_gives_check(board: Board, move: int) -> bool:
    board.push_legal(int(move))
    try:
        return bool(board.in_check_turn())
    finally:
        board.pop()


def _add_candidate(
    candidates: dict[int, Candidate],
    board: Board,
    move: int,
    source: str,
    detail: dict[str, Any] | None = None,
) -> None:
    move_i = int(move)
    if not bool(board.is_legal(move_i)):
        return
    cand = candidates.get(move_i)
    if cand is None:
        cand = Candidate(move=move_i, move_uci=internal_move_to_uci(move_i))
        candidates[move_i] = cand
    cand.sources.add(str(source))
    if detail:
        cand.source_details[str(source)] = detail


def _collect_tactical_candidates(
    board: Board,
    candidates: dict[int, Candidate],
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> None:
    root_stm_is_red = int(board.turn()) == 0
    for mv in [int(m) for m in board.legal_moves()]:
        if bool(board.is_capture(mv)):
            _add_candidate(candidates, board, mv, "legal_capture")
        if _move_gives_check(board, mv):
            _add_candidate(candidates, board, mv, "legal_check")
        if bool(board.is_legal(mv)):
            board.push_legal(mv)
            try:
                terminal_cp = _terminal_q_cp(
                    board,
                    root_stm_is_red,
                    max_plies=max_plies,
                    repeat_limit=repeat_limit,
                    repeat_min_ply=repeat_min_ply,
                    no_capture_limit=no_capture_limit,
                )
            finally:
                board.pop()
            if terminal_cp is not None and terminal_cp > 0.0:
                _add_candidate(candidates, board, mv, "immediate_terminal_win", {"root_q_cp": float(terminal_cp)})


def _collect_pika(pool: PikafishPool, jobs: list[PikafishJob], timeout_s: float) -> dict[int, PikafishResult]:
    pool.submit_all(jobs)
    try:
        results = pool.collect(len(jobs), timeout_s=float(timeout_s))
    except PikafishPoolTimeout as exc:
        results = exc.partial_results
    return {int(result.index): result for result in results}


def _rank(move_uci: str | None, ordered_moves: list[str]) -> int | None:
    if not move_uci:
        return None
    for idx, mv in enumerate(ordered_moves, start=1):
        if str(mv) == str(move_uci):
            return idx
    return None


def _avg(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def _med(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _position_sources(rows: list[dict[str, Any]], move_uci: str) -> set[str]:
    for row in rows:
        if str(row.get("move_uci")) == str(move_uci):
            return set(row.get("sources", []))
    return set()


def _source_detail(row: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if row is None:
        return {}
    details = row.get("source_details", {})
    if not isinstance(details, dict):
        return {}
    value = details.get(source, {})
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arena_json", nargs="+")
    parser.add_argument("--input-kind", choices=["arena", "audit"], default="arena")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--opening-suite-path", default="")
    parser.add_argument("--results", default="opp_win")
    parser.add_argument("--only-side", choices=["any", "red", "black"], default="black")
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--max-positions-per-file", type=int, default=0)
    parser.add_argument("--ply-stride", type=int, default=4)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--raw-policy-top-k", type=int, default=16)
    parser.add_argument("--mcts-top-k", type=int, default=16)
    parser.add_argument(
        "--selected-source",
        choices=["record", "mcts"],
        default="record",
        help=(
            "Which move to classify as the selected root move. 'record' preserves "
            "legacy behavior and uses the arena/audit move; 'mcts' uses the move "
            "chosen by the rerun MCTS with the current checkpoint and search args."
        ),
    )
    parser.add_argument("--pika-root-multipv", type=int, default=8)
    parser.add_argument("--regret-margin-cp", type=float, default=150.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-bf16", action="store_true")

    parser.add_argument("--mcts-sims", type=int, default=8000)
    parser.add_argument("--mcts-c-puct", type=float, default=1.45)
    parser.add_argument("--mcts-q-weight", type=float, default=1.0)
    parser.add_argument("--mcts-q-clip", type=float, default=1.0)
    parser.add_argument("--mcts-temperature-move", type=float, default=0.02)
    parser.add_argument("--mcts-seed", type=int, default=20260418)
    parser.add_argument("--mcts-tactical-mate1-extension", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mcts-tactical-mate2-extension", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--pika-root-depth", type=int, default=12)
    parser.add_argument("--pika-child-depth", type=int, default=14)
    parser.add_argument("--pika-workers", type=int, default=8)
    parser.add_argument("--pika-threads-per-worker", type=int, default=1)
    parser.add_argument("--pika-hash-mb", type=int, default=128)
    parser.add_argument("--pika-binary", default="/home/laure/pikafish/pikafish")
    parser.add_argument("--pika-timeout-s", type=float, default=7200.0)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    args = parser.parse_args()

    t0 = time.monotonic()
    paths = [Path(path) for path in args.arena_json]
    wanted = {part.strip() for part in str(args.results).split(",") if part.strip()}
    if str(args.input_kind) == "audit":
        positions = _extract_positions_from_audits(
            paths,
            results=wanted,
            only_side=str(args.only_side),
            max_positions=int(args.max_positions),
            max_positions_per_file=int(args.max_positions_per_file),
        )
    else:
        positions = _extract_positions(
            paths,
            results=wanted,
            only_side=str(args.only_side),
            max_positions=int(args.max_positions),
            max_positions_per_file=int(args.max_positions_per_file),
            ply_stride=int(args.ply_stride),
            min_ply=int(args.min_ply),
            max_plies=int(args.max_plies),
            opening_suite_path=str(args.opening_suite_path),
        )
    if not positions:
        raise SystemExit("no positions extracted")
    print(f"extracted {len(positions)} root positions from {len(paths)} arena JSON(s)", flush=True)

    device = torch.device(args.device)
    use_bfloat16 = not bool(args.disable_bf16)
    print(f"loading checkpoint {args.checkpoint}", flush=True)
    model, evaluator, checkpoint_step = _load_model_and_evaluator(Path(args.checkpoint), device, use_bfloat16)

    boards: list[Board] = []
    root_states: list[torch.Tensor] = []
    for pos in positions:
        board = _board_from_position(pos)
        boards.append(board)
        root_states.append(board.to_tensor_canonical().to(torch.float32)[0].contiguous())
    root_out = _forward(
        model,
        torch.stack(root_states, dim=0),
        device,
        batch_size=int(args.batch_size),
        use_bfloat16=use_bfloat16,
    )

    candidate_by_pos: list[dict[int, Candidate]] = []
    mcts_info: list[dict[str, Any]] = []
    root_jobs: list[PikafishJob] = []
    job_index = 0

    logged_mcts_count = sum(1 for pos in positions if pos.logged_search and pos.logged_search.get("root_stats"))
    print(
        f"rerunning V13 MCTS: positions={len(positions)} sims={int(args.mcts_sims)} "
        f"topK={int(args.mcts_top_k)} logged_root_stats={logged_mcts_count}",
        flush=True,
    )
    for i, (pos, board) in enumerate(zip(positions, boards)):
        candidates: dict[int, Candidate] = {}
        chosen_raw = int(uci_move_to_internal(str(pos.chosen_uci)[:4]))
        if str(args.selected_source) == "record":
            _add_candidate(candidates, board, chosen_raw, "selected")
        else:
            _add_candidate(candidates, board, chosen_raw, "record_selected")

        logged_search = pos.logged_search if isinstance(pos.logged_search, dict) else None
        if logged_search and logged_search.get("root_stats"):
            root_stats = [dict(row) for row in logged_search.get("root_stats", []) or []]
            root_stat_candidates = _root_stats_candidates(root_stats)
            best_uci = str(logged_search.get("root_best_move_uci", "") or logged_search.get("move_uci", ""))[:4]
            try:
                best_move = int(uci_move_to_internal(best_uci)) if best_uci else -1
            except Exception:
                best_move = -1
            idxs = []
            probs = []
            root_v = float(logged_search.get("root_value", 0.0) or 0.0)
        else:
            mcts_kwargs = dict(
                board=board,
                net=evaluator,
                num_simulations=int(args.mcts_sims),
                c_puct=float(args.mcts_c_puct),
                q_weight=float(args.mcts_q_weight),
                q_clip=float(args.mcts_q_clip),
                add_root_noise=False,
                dirichlet_alpha=0.3,
                dirichlet_eps=0.0,
                temperature_move=float(args.mcts_temperature_move),
                temperature_target=1.0,
                eval_batch_size=16,
                seed=int(args.mcts_seed) + i * 1009 + int(pos.ply),
                canonical_input=True,
                canonical_policy=True,
                max_plies=int(args.max_plies),
                repeat_limit=int(args.repeat_limit),
                repeat_min_ply=int(args.repeat_min_ply),
                no_capture_limit=int(args.no_capture_limit),
                tactical_mate1_extension=bool(args.mcts_tactical_mate1_extension),
                tactical_mate2_extension=bool(args.mcts_tactical_mate2_extension),
                c_puct_base=1.0,
                c_puct_factor=0.0,
                fpu_reduction_root=-1.0,
                fpu_reduction_tree=-1.0,
            )
            try:
                best_move, idxs, probs, root_v, root_stats = mcts_search_with_root_stats(**mcts_kwargs)
            except AttributeError:
                best_move, idxs, probs, root_v = mcts_search(**mcts_kwargs)
                root_stats = []
            root_stat_candidates = _root_stats_candidates(list(root_stats))
        if str(args.selected_source) == "mcts" and int(best_move) >= 0:
            _add_candidate(candidates, board, int(best_move), "selected")
        mcts_candidates = _root_policy_candidates_from_search(idxs, probs, board)
        root_stat_by_move = {int(mv): dict(row) for mv, row in root_stat_candidates}
        if root_stat_candidates:
            mcts_candidates = [
                (int(mv), float(row.get("visit_prob", 0.0)))
                for mv, row in root_stat_candidates
            ]
        for rank, (mv, prob) in enumerate(mcts_candidates[: max(1, int(args.mcts_top_k))], start=1):
            stat = root_stat_by_move.get(int(mv), {})
            detail = {
                "rank": int(rank),
                "visit_prob": float(prob),
                "visit_count": int(stat.get("visit_count", 0)),
                "target_prob": float(stat.get("target_prob", prob)),
                "prior": float(stat.get("prior", 0.0)),
                "q_root_pov": float(stat.get("q_root_pov", 0.0)),
                "q_child_pov": float(stat.get("q_child_pov", 0.0)),
                "ucb_score": float(stat.get("ucb_score", 0.0)),
                "canonical_idx": int(stat.get("canonical_idx", int(canonical_action(int(mv), bool(int(board.turn()) == 1))))),
            }
            _add_candidate(candidates, board, mv, "mcts_visit", detail)
        raw_policy = _legal_policy_topk(root_out["policy_logits"][i], board, int(args.raw_policy_top_k))
        for rank, (mv, prob) in enumerate(raw_policy, start=1):
            _add_candidate(candidates, board, mv, "raw_policy", {"rank": int(rank), "prob": float(prob)})
        _collect_tactical_candidates(
            board,
            candidates,
            max_plies=int(args.max_plies),
            repeat_limit=int(args.repeat_limit),
            repeat_min_ply=int(args.repeat_min_ply),
            no_capture_limit=int(args.no_capture_limit),
        )
        root_jobs.append(
            PikafishJob(
                index=job_index,
                fen=pos.fen,
                depth=int(args.pika_root_depth),
                multipv=int(args.pika_root_multipv),
            )
        )
        job_index += 1
        mcts_info.append(
            {
                "best_move_uci": internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else "",
                "root_value": float(root_v),
                "root_visit_candidates": [
                    {
                        "move_uci": internal_move_to_uci(int(mv)),
                        "visit_prob": float(prob),
                        "visit_count": int(root_stat_by_move.get(int(mv), {}).get("visit_count", 0)),
                        "prior": float(root_stat_by_move.get(int(mv), {}).get("prior", 0.0)),
                        "q_root_pov": float(root_stat_by_move.get(int(mv), {}).get("q_root_pov", 0.0)),
                        "ucb_score": float(root_stat_by_move.get(int(mv), {}).get("ucb_score", 0.0)),
                        "rank": int(rank),
                    }
                    for rank, (mv, prob) in enumerate(mcts_candidates[: max(1, int(args.mcts_top_k))], start=1)
                ],
                "root_stats": [
                    {
                        "move_uci": internal_move_to_uci(int(mv)),
                        "move_raw": int(mv),
                        "rank": int(rank),
                        "visit_count": int(row.get("visit_count", 0)),
                        "visit_prob": float(row.get("visit_prob", 0.0)),
                        "target_prob": float(row.get("target_prob", 0.0)),
                        "prior": float(row.get("prior", 0.0)),
                        "q_root_pov": float(row.get("q_root_pov", 0.0)),
                        "q_child_pov": float(row.get("q_child_pov", 0.0)),
                        "ucb_score": float(row.get("ucb_score", 0.0)),
                        "canonical_idx": int(row.get("canonical_idx", 0)),
                        "selected": bool(row.get("selected", False)),
                    }
                    for rank, (mv, row) in enumerate(root_stat_candidates, start=1)
                ],
                "root_stats_available": bool(root_stat_candidates),
            }
        )
        candidate_by_pos.append(candidates)

    print(
        f"querying Pikafish root MultiPV: positions={len(root_jobs)} "
        f"depth={int(args.pika_root_depth)} multipv={int(args.pika_root_multipv)}",
        flush=True,
    )
    pool = PikafishPool(
        num_workers=int(args.pika_workers),
        binary_path=str(args.pika_binary),
        threads_per_worker=int(args.pika_threads_per_worker),
        hash_mb=int(args.pika_hash_mb),
    )
    try:
        root_results_by_job = _collect_pika(pool, root_jobs, timeout_s=float(args.pika_timeout_s))
    finally:
        pool.close()

    root_pika: list[dict[str, Any] | None] = [None for _ in positions]
    for i, job in enumerate(root_jobs):
        result = root_results_by_job.get(int(job.index))
        if result is None or result.error:
            continue
        multipv_rows = [
            {"move_uci": str(move)[:4], "score_cp": int(cp), "rank": int(rank)}
            for rank, (move, cp) in enumerate(result.multipv_moves or [], start=1)
        ]
        if not multipv_rows and result.best_move:
            multipv_rows.append({"move_uci": str(result.best_move)[:4], "score_cp": int(result.eval_cp), "rank": 1})
        root_pika[i] = {
            "best_move_uci": str(result.best_move)[:4],
            "eval_cp_root_pov": int(result.eval_cp),
            "mate_in": result.mate_in,
            "multipv": multipv_rows,
        }
        board = boards[i]
        for row in multipv_rows:
            try:
                mv = int(uci_move_to_internal(str(row["move_uci"])[:4]))
            except Exception:
                continue
            _add_candidate(
                candidate_by_pos[i],
                board,
                mv,
                "pika_multipv",
                {"rank": int(row["rank"]), "root_score_cp": int(row["score_cp"])},
            )

    child_jobs: list[PikafishJob] = []
    child_job_refs: dict[int, tuple[int, int]] = {}
    candidate_rows_by_pos: list[list[dict[str, Any]]] = []
    child_states: list[torch.Tensor] = []
    child_refs: list[tuple[int, int]] = []

    for pos_i, (pos, board, candidates) in enumerate(zip(positions, boards, candidate_by_pos)):
        rows: list[dict[str, Any]] = []
        root_stm_is_red = int(board.turn()) == 0
        ordered = sorted(candidates.values(), key=lambda c: (0 if "selected" in c.sources else 1, c.move_uci))
        for cand in ordered:
            child = Board()
            child = _board_from_position(pos)
            if not bool(child.is_legal(int(cand.move))):
                continue
            is_capture = bool(child.is_capture(int(cand.move)))
            child.push_legal(int(cand.move))
            terminal_cp = _terminal_q_cp(
                child,
                root_stm_is_red,
                max_plies=int(args.max_plies),
                repeat_limit=int(args.repeat_limit),
                repeat_min_ply=int(args.repeat_min_ply),
                no_capture_limit=int(args.no_capture_limit),
            )
            row_i = len(rows)
            row = {
                "move": int(cand.move),
                "move_uci": cand.move_uci,
                "sources": sorted(cand.sources),
                "source_details": cand.source_details,
                "is_selected": "selected" in cand.sources,
                "is_capture": bool(is_capture),
                "gives_check": "legal_check" in cand.sources,
                "terminal_q_root_pov_cp": terminal_cp,
            }
            rows.append(row)
            child_states.append(child.to_tensor_canonical().to(torch.float32)[0].contiguous())
            child_refs.append((pos_i, row_i))
            if terminal_cp is None:
                child_jobs.append(
                    PikafishJob(
                        index=job_index,
                        fen=_pad_fen(child.fen()),
                        depth=int(args.pika_child_depth),
                        multipv=1,
                    )
                )
                child_job_refs[job_index] = (pos_i, row_i)
                job_index += 1
        candidate_rows_by_pos.append(rows)

    print(f"model child evals: {len(child_states)} candidate children", flush=True)
    child_out = _forward(
        model,
        torch.stack(child_states, dim=0),
        device,
        batch_size=int(args.batch_size),
        use_bfloat16=use_bfloat16,
    )
    for child_i, (pos_i, row_i) in enumerate(child_refs):
        child_v_opp = float(child_out["value_scalar"][child_i].flatten()[0].item())
        candidate_rows_by_pos[pos_i][row_i]["model_child_value_opponent_pov"] = child_v_opp
        candidate_rows_by_pos[pos_i][row_i]["model_q_root_pov"] = -child_v_opp

    print(
        f"querying Pikafish child evals: children={len(child_jobs)} "
        f"depth={int(args.pika_child_depth)} workers={int(args.pika_workers)}",
        flush=True,
    )
    pool = PikafishPool(
        num_workers=int(args.pika_workers),
        binary_path=str(args.pika_binary),
        threads_per_worker=int(args.pika_threads_per_worker),
        hash_mb=int(args.pika_hash_mb),
    )
    try:
        child_results_by_job = _collect_pika(pool, child_jobs, timeout_s=float(args.pika_timeout_s))
    finally:
        pool.close()

    for job_id, (pos_i, row_i) in child_job_refs.items():
        result = child_results_by_job.get(int(job_id))
        if result is None or result.error:
            candidate_rows_by_pos[pos_i][row_i]["pika_missing"] = True
            continue
        candidate_rows_by_pos[pos_i][row_i]["pika_child_eval_opponent_pov_cp"] = int(result.eval_cp)
        candidate_rows_by_pos[pos_i][row_i]["pika_q_root_pov_cp"] = int(-result.eval_cp)
        candidate_rows_by_pos[pos_i][row_i]["pika_mate_in_child"] = result.mate_in
    for rows in candidate_rows_by_pos:
        for row in rows:
            if row.get("terminal_q_root_pov_cp") is not None:
                row["pika_child_eval_opponent_pov_cp"] = int(-float(row["terminal_q_root_pov_cp"]))
                row["pika_q_root_pov_cp"] = float(row["terminal_q_root_pov_cp"])
                row["pika_mate_in_child"] = None

    records: list[dict[str, Any]] = []
    scored = 0
    missing_candidate = 0
    ranking_failure = 0
    verifier_candidate_failure = 0
    deep_topk_verifier_failure = 0
    pika_root_horizon_failure = 0
    pika_root_would_replace = 0
    catastrophic_failure = 0
    q_inversion_failure = 0
    prior_visit_failure = 0
    selected_regrets: list[float] = []
    mcts_best_regrets: list[float] = []
    model_q_best_regrets: list[float] = []
    candidate_counts: list[float] = []
    top_source_counts: dict[str, int] = {}

    for i, pos in enumerate(positions):
        rows = [row for row in candidate_rows_by_pos[i] if "pika_q_root_pov_cp" in row]
        if not rows:
            continue
        scored += 1
        rows_by_teacher = sorted(rows, key=lambda row: float(row["pika_q_root_pov_cp"]), reverse=True)
        rows_by_model_q = sorted(rows, key=lambda row: float(row.get("model_q_root_pov", -math.inf)), reverse=True)
        best = rows_by_teacher[0]
        best_cp = float(best["pika_q_root_pov_cp"])
        selected = next((row for row in rows if bool(row.get("is_selected"))), None)
        selected_cp = None if selected is None else float(selected["pika_q_root_pov_cp"])
        selected_regret = None if selected_cp is None else best_cp - selected_cp
        if selected_regret is not None:
            selected_regrets.append(float(selected_regret))

        selected_root_score = None
        best_root_score = None
        if selected is not None:
            selected_root_score = selected.get("source_details", {}).get("pika_multipv", {}).get("root_score_cp")
        best_root_score = best.get("source_details", {}).get("pika_multipv", {}).get("root_score_cp")
        shallow_root_regret = None
        if selected_root_score is not None and best_root_score is not None:
            shallow_root_regret = float(best_root_score) - float(selected_root_score)
        selected_mcts = _source_detail(selected, "mcts_visit")
        best_mcts = _source_detail(best, "mcts_visit")
        selected_mcts_q = selected_mcts.get("q_root_pov")
        best_mcts_q = best_mcts.get("q_root_pov")
        selected_visit_prob = selected_mcts.get("visit_prob")
        best_visit_prob = best_mcts.get("visit_prob")
        selected_prior = selected_mcts.get("prior")
        best_prior = best_mcts.get("prior")

        mcts_best_uci = str(mcts_info[i].get("best_move_uci", "") or "")
        mcts_best = next((row for row in rows if str(row.get("move_uci")) == mcts_best_uci), None)
        if mcts_best is not None:
            mcts_best_regrets.append(best_cp - float(mcts_best["pika_q_root_pov_cp"]))
        model_q_best = rows_by_model_q[0] if rows_by_model_q else None
        if model_q_best is not None:
            model_q_best_regrets.append(best_cp - float(model_q_best["pika_q_root_pov_cp"]))

        best_sources = set(best.get("sources", []))
        for source in best_sources:
            top_source_counts[source] = int(top_source_counts.get(source, 0)) + 1
        v13_candidate_sources = {"mcts_visit", "raw_policy", "selected"}
        best_in_v13_candidates = bool(best_sources & v13_candidate_sources)
        best_in_mcts_topk = "mcts_visit" in best_sources
        best_in_raw_policy = "raw_policy" in best_sources
        selected_bad = bool(selected_regret is not None and selected_regret >= float(args.regret_margin_cp))
        selected_mate_bad = bool(
            selected is not None
            and selected.get("pika_mate_in_child") is not None
            and selected_cp is not None
            and selected_cp < 0.0
        )
        is_catastrophic = bool(
            selected_bad
            and (
                (selected_regret is not None and selected_regret >= 1000.0)
                or selected_mate_bad
            )
        )
        is_missing = bool(selected_bad and not best_in_v13_candidates)
        is_ranking = bool(selected_bad and best_in_v13_candidates)
        is_deep_topk_verifier = bool(selected_bad and best_in_mcts_topk)
        is_verifier_candidate = bool(selected_bad and not best_in_mcts_topk and ("pika_multipv" in best_sources or "legal_check" in best_sources or "legal_capture" in best_sources))
        is_horizon = bool(selected_bad and shallow_root_regret is not None and float(shallow_root_regret) < float(args.regret_margin_cp))
        is_root_pika_replace = bool(selected_bad and shallow_root_regret is not None and float(shallow_root_regret) >= float(args.regret_margin_cp))
        is_q_inversion = bool(
            selected_bad
            and selected_mcts_q is not None
            and best_mcts_q is not None
            and float(best_mcts_q) < float(selected_mcts_q)
        )
        is_prior_visit = bool(
            selected_bad
            and not is_q_inversion
            and selected_mcts_q is not None
            and best_mcts_q is not None
            and float(best_mcts_q) >= float(selected_mcts_q) - 0.02
            and (
                (
                    selected_visit_prob is not None
                    and best_visit_prob is not None
                    and float(selected_visit_prob) >= max(0.05, 2.0 * float(best_visit_prob))
                )
                or (
                    selected_prior is not None
                    and best_prior is not None
                    and float(selected_prior) >= max(0.02, 2.0 * float(best_prior))
                )
            )
        )
        if is_catastrophic:
            catastrophic_failure += 1
        if is_missing:
            missing_candidate += 1
        if is_ranking:
            ranking_failure += 1
        if is_deep_topk_verifier:
            deep_topk_verifier_failure += 1
        if is_verifier_candidate:
            verifier_candidate_failure += 1
        if is_horizon:
            pika_root_horizon_failure += 1
        if is_root_pika_replace:
            pika_root_would_replace += 1
        if is_q_inversion:
            q_inversion_failure += 1
        if is_prior_visit:
            prior_visit_failure += 1
        candidate_counts.append(float(len(rows)))

        mcts_q_order = [str(row["move_uci"]) for row in sorted(
            rows,
            key=lambda row: float(row.get("source_details", {}).get("mcts_visit", {}).get("q_root_pov", -math.inf)),
            reverse=True,
        ) if "mcts_visit" in row.get("sources", [])]
        mcts_order = [str(row["move_uci"]) for row in sorted(
            rows,
            key=lambda row: int(row.get("source_details", {}).get("mcts_visit", {}).get("rank", 999999)),
        ) if "mcts_visit" in row.get("sources", [])]
        raw_order = [str(row["move_uci"]) for row in sorted(
            rows,
            key=lambda row: int(row.get("source_details", {}).get("raw_policy", {}).get("rank", 999999)),
        ) if "raw_policy" in row.get("sources", [])]
        pika_order = [str(row["move_uci"]) for row in sorted(
            rows,
            key=lambda row: int(row.get("source_details", {}).get("pika_multipv", {}).get("rank", 999999)),
        ) if "pika_multipv" in row.get("sources", [])]

        records.append(
            {
                "position": {
                    "fen": pos.fen,
                    "ply": pos.ply,
                    "search_plies": pos.search_plies,
                    "no_capture_count": pos.no_capture_count,
                    "repetition_count_hint": pos.repetition_count_hint,
                    "game_index": pos.game_index,
                    "source_arena": pos.source_arena,
                    "result": pos.result,
                    "termination": pos.termination,
                    "our_side": pos.our_side,
                    "chosen_uci": pos.chosen_uci,
                    "opening_id": pos.opening_id,
                    "opening_index": pos.opening_index,
                    "opening_fen": pos.opening_fen,
                },
                "root_model_value": float(root_out["value_scalar"][i].flatten()[0].item()),
                "mcts": mcts_info[i],
                "root_pika": root_pika[i],
                "teacher_best": {
                    "move_uci": str(best["move_uci"]),
                    "pika_q_root_pov_cp": best_cp,
                    "sources": sorted(best_sources),
                    "mcts_rank": _rank(str(best["move_uci"]), mcts_order),
                    "mcts_q_rank": _rank(str(best["move_uci"]), mcts_q_order),
                    "raw_policy_rank": _rank(str(best["move_uci"]), raw_order),
                    "pika_multipv_rank": _rank(str(best["move_uci"]), pika_order),
                    "model_q_rank": _rank(str(best["move_uci"]), [str(row["move_uci"]) for row in rows_by_model_q]),
                },
                "selected": None if selected is None else {
                    "move_uci": str(selected["move_uci"]),
                    "pika_q_root_pov_cp": float(selected["pika_q_root_pov_cp"]),
                    "regret_cp": selected_regret,
                    "mcts_rank": _rank(str(selected["move_uci"]), mcts_order),
                    "mcts_q_rank": _rank(str(selected["move_uci"]), mcts_q_order),
                    "mcts_q_root_pov": _source_detail(selected, "mcts_visit").get("q_root_pov"),
                    "mcts_prior": _source_detail(selected, "mcts_visit").get("prior"),
                    "mcts_visit_prob": _source_detail(selected, "mcts_visit").get("visit_prob"),
                    "sources": sorted(_position_sources(rows, str(selected["move_uci"]))),
                },
                "classification": {
                    "bad_root": selected_bad,
                    "catastrophic": is_catastrophic,
                    "selected_regret_cp": selected_regret,
                    "pika_root_score_regret_cp": shallow_root_regret,
                    "best_in_v13_candidates": best_in_v13_candidates,
                    "best_in_mcts_topk": best_in_mcts_topk,
                    "best_in_raw_policy_topk": best_in_raw_policy,
                    "missing_candidate_failure": is_missing,
                    "ranking_failure": is_ranking,
                    "deep_topk_verifier_would_help": is_deep_topk_verifier,
                    "candidate_union_verifier_would_help": is_verifier_candidate,
                    "pika_root_horizon_failure": is_horizon,
                    "pika_root_would_have_replaced": is_root_pika_replace,
                    "root_verifier_detectable": is_root_pika_replace,
                    "q_inversion": is_q_inversion,
                    "prior_visit_failure": is_prior_visit,
                },
                "candidate_rows": sorted(rows, key=lambda row: float(row["pika_q_root_pov_cp"]), reverse=True),
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "arena_json": [str(path) for path in paths],
        "positions_extracted": len(positions),
        "positions_scored": scored,
        "raw_policy_top_k": int(args.raw_policy_top_k),
        "mcts_top_k": int(args.mcts_top_k),
        "selected_source": str(args.selected_source),
        "mcts_sims": int(args.mcts_sims),
        "pika_root_depth": int(args.pika_root_depth),
        "pika_child_depth": int(args.pika_child_depth),
        "pika_root_multipv": int(args.pika_root_multipv),
        "regret_margin_cp": float(args.regret_margin_cp),
        "failure_rates": {
            "bad_root": (ranking_failure + missing_candidate) / scored if scored else None,
            "catastrophic": catastrophic_failure / scored if scored else None,
            "missing_candidate": missing_candidate / scored if scored else None,
            "ranking": ranking_failure / scored if scored else None,
            "q_inversion": q_inversion_failure / scored if scored else None,
            "prior_visit_failure": prior_visit_failure / scored if scored else None,
            "deep_topk_verifier_would_help": deep_topk_verifier_failure / scored if scored else None,
            "candidate_union_verifier_would_help": verifier_candidate_failure / scored if scored else None,
            "pika_root_horizon_failure": pika_root_horizon_failure / scored if scored else None,
            "pika_root_would_have_replaced": pika_root_would_replace / scored if scored else None,
            "root_verifier_detectable": pika_root_would_replace / scored if scored else None,
        },
        "counts": {
            "bad_root": ranking_failure + missing_candidate,
            "catastrophic": catastrophic_failure,
            "missing_candidate": missing_candidate,
            "ranking": ranking_failure,
            "q_inversion": q_inversion_failure,
            "prior_visit_failure": prior_visit_failure,
            "deep_topk_verifier_would_help": deep_topk_verifier_failure,
            "candidate_union_verifier_would_help": verifier_candidate_failure,
            "pika_root_horizon_failure": pika_root_horizon_failure,
            "pika_root_would_have_replaced": pika_root_would_replace,
            "root_verifier_detectable": pika_root_would_replace,
        },
        "regret_cp": {
            "selected_mean": _avg(selected_regrets),
            "selected_median": _med(selected_regrets),
            "mcts_best_mean": _avg(mcts_best_regrets),
            "model_q_best_mean": _avg(model_q_best_regrets),
        },
        "candidate_count": {
            "mean": _avg(candidate_counts),
            "median": _med(candidate_counts),
            "max": None if not candidate_counts else int(max(candidate_counts)),
        },
        "teacher_best_source_counts": dict(sorted(top_source_counts.items())),
        "elapsed_s": time.monotonic() - t0,
    }

    payload = {
        "summary": summary,
        "records": records,
        "root_regret_schema_hint": {
            "state": "position.fen",
            "candidate_move": "candidate_rows[].move_uci",
            "v13_prior": "candidate_rows[].source_details.raw_policy.prob",
            "v13_visit_prob": "candidate_rows[].source_details.mcts_visit.visit_prob",
            "teacher_child_eval": "candidate_rows[].pika_q_root_pov_cp",
            "teacher_best_eval": "max(candidate_rows[].pika_q_root_pov_cp)",
            "regret_cp": "teacher_best_eval - teacher_child_eval",
            "is_refuted": f"regret_cp > {float(args.regret_margin_cp)}",
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# V13 Root Decision Audit",
        "",
        f"- positions scored: {scored}/{len(positions)}",
        f"- checkpoint step: {checkpoint_step}",
        f"- MCTS: {int(args.mcts_sims)} sims, top{int(args.mcts_top_k)}",
        f"- raw policy: top{int(args.raw_policy_top_k)}",
        f"- Pikafish: root d{int(args.pika_root_depth)} MultiPV {int(args.pika_root_multipv)}, child d{int(args.pika_child_depth)}",
        f"- elapsed: {summary['elapsed_s']:.1f}s",
        "",
        "## Failure Split",
        "",
        f"- bad root: {ranking_failure + missing_candidate}/{scored} ({_fmt_pct(summary['failure_rates']['bad_root'])})",
        f"- catastrophic: {catastrophic_failure}/{scored} ({_fmt_pct(summary['failure_rates']['catastrophic'])})",
        f"- missing candidate: {missing_candidate}/{scored} ({_fmt_pct(summary['failure_rates']['missing_candidate'])})",
        f"- ranking failure: {ranking_failure}/{scored} ({_fmt_pct(summary['failure_rates']['ranking'])})",
        f"- Q inversion: {q_inversion_failure}/{scored} ({_fmt_pct(summary['failure_rates']['q_inversion'])})",
        f"- prior/visit failure: {prior_visit_failure}/{scored} ({_fmt_pct(summary['failure_rates']['prior_visit_failure'])})",
        f"- deep top-K verifier would help: {deep_topk_verifier_failure}/{scored} ({_fmt_pct(summary['failure_rates']['deep_topk_verifier_would_help'])})",
        f"- candidate-union verifier would help: {verifier_candidate_failure}/{scored} ({_fmt_pct(summary['failure_rates']['candidate_union_verifier_would_help'])})",
        f"- Pika root horizon failure: {pika_root_horizon_failure}/{scored} ({_fmt_pct(summary['failure_rates']['pika_root_horizon_failure'])})",
        f"- Pika root would have replaced: {pika_root_would_replace}/{scored} ({_fmt_pct(summary['failure_rates']['pika_root_would_have_replaced'])})",
        "",
        "## Regret",
        "",
        f"- selected mean regret cp: {_fmt_num(summary['regret_cp']['selected_mean'])}",
        f"- selected median regret cp: {_fmt_num(summary['regret_cp']['selected_median'])}",
        f"- MCTS best mean regret cp: {_fmt_num(summary['regret_cp']['mcts_best_mean'])}",
        f"- model-Q best mean regret cp: {_fmt_num(summary['regret_cp']['model_q_best_mean'])}",
        "",
        "## Candidate Sources For Teacher Best",
        "",
    ]
    for source, count in sorted(top_source_counts.items()):
        md_lines.append(f"- {source}: {count}")
    md_lines.extend(["", "## Worst Selected-Regret Positions", ""])
    worst = sorted(
        [rec for rec in records if rec["selected"] is not None],
        key=lambda rec: float(rec["selected"].get("regret_cp") or 0.0),
        reverse=True,
    )[:10]
    for rec in worst:
        selected = rec["selected"] or {}
        best = rec["teacher_best"]
        pos = rec["position"]
        cls = rec["classification"]
        tags = []
        if cls["missing_candidate_failure"]:
            tags.append("missing")
        if cls["ranking_failure"]:
            tags.append("ranking")
        if cls["candidate_union_verifier_would_help"]:
            tags.append("union-verifier")
        md_lines.append(
            f"- game {pos['game_index']} ply {pos['ply']} {pos['opening_id']}: "
            f"selected {selected.get('move_uci')} vs best {best['move_uci']} "
            f"regret {float(selected.get('regret_cp') or 0.0):.0f} cp "
            f"sources={','.join(best['sources'])} tags={','.join(tags) or 'none'}"
        )

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
