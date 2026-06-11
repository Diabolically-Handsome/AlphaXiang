"""External arena: our model (via MCTS) vs Pikafish at configurable strength.

Produces an Elo estimate and saves a JSON summary + per-game record with move lists
(so we can later distill Pikafish's moves as policy targets).

Example:
    python tools/external_arena.py \
        --checkpoint /home/laure/alphaxiang/training_runs/run_001/best.pt \
        --games 10 \
        --opp-depth 5 \
        --our-sims 400 \
        --output-dir /home/laure/alphaxiang/external_arena_runs

The harness keeps a single Pikafish subprocess per match (re-using across games via
`ucinewgame`) and loads our model once into GPU memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS = _REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import random  # noqa: E402

from alpha_beta_search import AlphaBetaConfig, alpha_beta_search  # noqa: E402
from cross_game_batcher import CrossGameBatcher  # noqa: E402
from elephantart_opponent import make_elephantart_opponent  # noqa: E402
from fairy_stockfish_opponent import make_fairy_stockfish_opponent  # noqa: E402
from pikafish_opponent import (  # noqa: E402
    PikafishOpponent,
    internal_move_to_uci,
    uci_move_to_internal,
)
from xiangqi_mcts_ext import (  # noqa: E402
    Board,
    canonical_action,
    make_gpu_evaluator,
    mcts_search,
    mcts_search_with_root_stats,
)
from xiangqi_danger_model import load_danger_checkpoint  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


# ----- Xiangqi terminal codes (mirror of TERMINAL_* enums in the C++ module) -----
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


def _pad_fen(fen: str) -> str:
    parts = fen.strip().split()
    # Standard UCI Xiangqi FEN has 6 fields: placement, side, castling, ep, halfmove, fullmove.
    # Our cpp emits 2 (placement + side); pad defensively.
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


def _load_opening_suite(path: str | Path, *, max_openings: int = 0) -> list[dict[str, Any]]:
    suite_path = Path(path)
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    raw_entries = payload.get("positions", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        raise ValueError(f"opening suite must be a list or contain a positions list: {suite_path}")
    openings: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"opening suite entry #{idx + 1} must be an object")
        fen = str(raw.get("fen", "")).strip()
        if not fen:
            raise ValueError(f"opening suite entry #{idx + 1} is missing fen")
        entry = dict(raw)
        entry["id"] = str(raw.get("id", f"opening_{idx + 1:03d}"))
        entry["fen"] = _pad_fen(fen)
        entry["plies"] = int(raw.get("plies", 0))
        entry["no_capture_count"] = int(raw.get("no_capture_count", 0))
        entry["repetition_count_hint"] = int(raw.get("repetition_count_hint", 1))

        board = Board()
        board.set_fen(str(entry["fen"]))
        board.set_search_context(
            int(entry["plies"]),
            int(entry["no_capture_count"]),
            int(entry["repetition_count_hint"]),
        )
        if not board.legal_moves():
            raise ValueError(f"opening suite entry has no legal moves: {entry['id']} {entry['fen']}")
        openings.append(entry)
        if int(max_openings) > 0 and len(openings) >= int(max_openings):
            break
    if not openings:
        raise ValueError(f"opening suite is empty: {suite_path}")
    return openings


def _initialize_opening(board: Board, opening_entry: dict[str, Any] | None) -> None:
    if opening_entry is None:
        return
    board.set_fen(str(opening_entry["fen"]))
    board.set_search_context(
        int(opening_entry.get("plies", 0)),
        int(opening_entry.get("no_capture_count", 0)),
        int(opening_entry.get("repetition_count_hint", 1)),
    )


@dataclass
class GameRecord:
    index: int
    our_side: str  # "red" or "black"
    moves_uci: list[str] = field(default_factory=list)
    result: str = ""  # "our_win", "opp_win", "draw"
    termination: str = ""
    plies: int = 0
    opening_fen: str = ""
    opening_id: str = ""
    opening_index: int | None = None
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    search_stats: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ArenaResult:
    checkpoint: str
    games: int
    opp_depth: int | None
    opp_movetime_ms: int | None
    opp_nodes: int | None
    opp_uci_elo: int | None
    opp_uci_limit_strength: bool
    our_sims: int
    our_wins: int = 0
    opp_wins: int = 0
    draws: int = 0
    termination_counts: dict[str, int] = field(default_factory=dict)
    avg_plies: float = 0.0
    our_side_counts: dict[str, int] = field(default_factory=dict)
    our_wins_as_red: int = 0
    our_wins_as_black: int = 0
    score_rate: float = 0.0
    elo_estimate: float | None = None
    per_game: list[GameRecord] = field(default_factory=list)
    search_kind: str = "mcts"
    search_stats_summary: dict[str, Any] = field(default_factory=dict)
    shadow_value_summary: dict[str, Any] = field(default_factory=dict)
    symbolic_guard_summary: dict[str, Any] = field(default_factory=dict)


def _load_model(checkpoint_path: Path, device: torch.device, use_bfloat16: bool = True):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    evaluator = make_gpu_evaluator(model, device=str(device), use_bfloat16=use_bfloat16)
    step = int(state.get("global_step", 0))
    return evaluator, step


class _ChimeraModel(torch.nn.Module):
    """Policy head from one checkpoint, value (and wdl) head from another.

    Used to externally validate spliced players (e.g. self-play policy + geo
    value) against engine ladders. Both submodels run a full forward; only the
    head outputs are recombined — identical to the chimera gate harness.
    """

    def __init__(self, policy_model: torch.nn.Module, value_model: torch.nn.Module) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, x):
        out_p = self.policy_model(x)
        out_v = self.value_model(x)
        out = dict(out_p)
        out["value_scalar"] = out_v["value_scalar"]
        if "wdl_logits" in out_v:
            out["wdl_logits"] = out_v["wdl_logits"]
        return out


def _build_arena_model(checkpoint_path: Path, value_checkpoint_path: str | None):
    """Build the candidate model, optionally as a policy/value chimera."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    step = int(state.get("global_step", 0))
    if value_checkpoint_path:
        v_state = torch.load(Path(value_checkpoint_path), map_location="cpu", weights_only=False)
        v_model = build_model_from_checkpoint_state(v_state)
        model = _ChimeraModel(model, v_model)
        print(
            f"[external-arena] CHIMERA mode: policy={checkpoint_path} value={value_checkpoint_path}",
            flush=True,
        )
    return model, step


class _ValueSourceEvaluator:
    """Small adapter for search-side ablations.

    The C++ MCTS consumes ``value_scalar``.  For WDL-Q experiments we keep the
    model and MCTS unchanged, but replace that scalar with P(win)-P(loss) derived
    from the existing WDL head.  Default scalar behavior is untouched.
    """

    def __init__(self, base, value_source: str) -> None:
        self.base = base
        self.value_source = str(value_source)

    def __call__(self, batch_cpu: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(batch_cpu)
        if self.value_source == "scalar":
            return out
        if self.value_source != "wdl":
            raise ValueError(f"unknown value_source={self.value_source!r}")
        wdl_logits = out.get("wdl_logits")
        if wdl_logits is None:
            raise KeyError("--our-value-source=wdl requires model output 'wdl_logits'")
        probs = torch.softmax(wdl_logits.float(), dim=-1)
        out = dict(out)
        out["value_scalar"] = (probs[:, 0:1] - probs[:, 2:3]).to(dtype=torch.float32).contiguous()
        return out

    def stats(self):
        if hasattr(self.base, "stats"):
            return self.base.stats()
        raise AttributeError(f"{type(self.base).__name__!s} has no stats()")

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()


def _wrap_value_source(evaluator, value_source: str):
    value_source = str(value_source)
    if value_source == "scalar":
        return evaluator
    return _ValueSourceEvaluator(evaluator, value_source)


class _DangerHeadRuntime:
    def __init__(self, checkpoint: str | Path, *, device: torch.device, use_bfloat16: bool = True) -> None:
        self.device = torch.device(device)
        self.use_bfloat16 = bool(use_bfloat16 and self.device.type == "cuda")
        self.model = load_danger_checkpoint(str(checkpoint), map_location="cpu")
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def score_moves(self, board: Board, candidates: list[tuple[float, int]]) -> dict[int, float]:
        states: list[torch.Tensor] = []
        moves: list[int] = []
        legal = {int(move) for move in board.legal_moves()}
        for _prob, move in candidates:
            raw_move = int(move)
            if raw_move not in legal:
                continue
            board.push(raw_move)
            try:
                state = board.to_tensor_canonical().to(torch.float32)[0].contiguous()
            finally:
                board.pop()
            states.append(state)
            moves.append(raw_move)
        if not states:
            return {}
        batch = torch.stack(states, dim=0).to(device=self.device, dtype=torch.float32)
        with self.lock:
            if self.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model(batch)
            else:
                out = self.model(batch)
        risks = torch.sigmoid(out["risk_logit"].float()).detach().cpu().tolist()
        return {move: float(risk) for move, risk in zip(moves, risks)}


def _apply_root_danger_head_rerank(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    danger_runtime: _DangerHeadRuntime,
    top_k: int,
    danger_lambda: float,
    veto_threshold: float,
) -> tuple[int, dict[str, Any] | None]:
    if int(best_move) < 0:
        return int(best_move), None
    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    candidates = candidates[: max(1, int(top_k))]
    risk_by_move = danger_runtime.score_moves(board, candidates)
    if not risk_by_move:
        return int(best_move), None

    candidate_rows: list[dict[str, Any]] = []
    best_score = -1e30
    replacement = int(best_move)
    for prob, move in candidates:
        move = int(move)
        risk = float(risk_by_move.get(move, 1.0))
        log_prior = math.log(max(float(prob), 1e-12))
        score = log_prior - float(danger_lambda) * risk
        vetoed = risk >= float(veto_threshold)
        candidate_rows.append(
            {
                "move_uci": internal_move_to_uci(move),
                "prob": float(prob),
                "danger": risk,
                "adjusted_score": float(score),
                "vetoed": bool(vetoed),
            }
        )
        if vetoed:
            continue
        if score > best_score:
            best_score = score
            replacement = move

    if replacement == int(best_move):
        return int(best_move), None
    return replacement, {
        "guard_type": "root_cnn_danger_rerank",
        "reason": "cnn_danger_lowered_selected_root_candidate",
        "original_move_uci": internal_move_to_uci(int(best_move)),
        "replacement_move_uci": internal_move_to_uci(int(replacement)),
        "original_prob": next((float(prob) for prob, move in candidates if int(move) == int(best_move)), None),
        "replacement_prob": next((float(prob) for prob, move in candidates if int(move) == int(replacement)), None),
        "original_danger": float(risk_by_move.get(int(best_move), 1.0)),
        "replacement_danger": float(risk_by_move.get(int(replacement), 1.0)),
        "danger_lambda": float(danger_lambda),
        "veto_threshold": float(veto_threshold),
        "top_candidates": candidate_rows[: int(top_k)],
    }


def _apply_root_danger_head_triage(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    danger_runtime: _DangerHeadRuntime,
    top_k: int,
    triage_threshold: float,
    exact_plies: int,
    exact_max_candidates: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None]:
    if int(best_move) < 0:
        return int(best_move), None
    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    candidates = candidates[: max(1, int(top_k))]
    if all(int(move) != int(best_move) for _prob, move in candidates):
        candidates.insert(0, (1.0, int(best_move)))
    risk_by_move = danger_runtime.score_moves(board, candidates)
    if not risk_by_move:
        return int(best_move), None
    best_risk = float(risk_by_move.get(int(best_move), 1.0))
    if best_risk < float(triage_threshold):
        return int(best_move), None

    event_base: dict[str, Any] = {
        "guard_type": "root_cnn_danger_triage",
        "reason": "cnn_danger_triggered_exact_guard",
        "original_move_uci": internal_move_to_uci(int(best_move)),
        "replacement_move_uci": internal_move_to_uci(int(best_move)),
        "original_danger": best_risk,
        "replacement_danger": best_risk,
        "triage_threshold": float(triage_threshold),
        "exact_plies": int(exact_plies),
    }

    checked_move = int(best_move)
    exact_event: dict[str, Any] | None = None
    checked_move, exact_event = _apply_root_mate1_blunder_guard(
        board,
        checked_move,
        idxs,
        probs,
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    )
    if exact_event is None and int(exact_plies) >= 3:
        checked_move, exact_event = _apply_root_mate2_blunder_guard(
            board,
            checked_move,
            idxs,
            probs,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
    if exact_event is None and int(exact_plies) >= 5:
        checked_move, exact_event = _apply_root_forcing_check_blunder_guard(
            board,
            checked_move,
            idxs,
            probs,
            plies_remaining=int(exact_plies),
            max_candidates=int(exact_max_candidates),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )

    if exact_event is None:
        event_base["reason"] = "cnn_danger_triggered_but_exact_guard_found_no_refutation"
        return int(best_move), event_base

    event_base["reason"] = "cnn_danger_triggered_exact_guard_replaced_move"
    event_base["replacement_move_uci"] = str(exact_event.get("replacement_move_uci", event_base["replacement_move_uci"]))
    replacement_uci = str(event_base["replacement_move_uci"])
    for _prob, move in candidates:
        if internal_move_to_uci(int(move)) == replacement_uci:
            event_base["replacement_danger"] = float(risk_by_move.get(int(move), 1.0))
            break
    event_base["exact_guard_event"] = exact_event
    return int(checked_move), event_base


def _move_is_terminal_win_for_mover(
    board: Board,
    move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
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
            return False
        red_result = int(board.terminal_result_red_view(term))
        return red_result != 0 and ((red_result > 0) == mover_is_red)
    finally:
        board.pop()


def _side_to_move_has_mate1(
    board: Board,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    for reply in board.legal_moves():
        if _move_is_terminal_win_for_mover(
            board,
            int(reply),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ):
            return True
    return False


def _move_allows_opponent_mate1(
    board: Board,
    move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    board.push(int(move))
    try:
        term = int(board.terminal_code(
            int(max_plies),
            int(repeat_limit),
            int(repeat_min_ply),
            int(no_capture_limit),
        ))
        if term != TERMINAL_ONGOING:
            return False
        return _side_to_move_has_mate1(
            board,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
    finally:
        board.pop()


def _terminal_is_win_for_side_to_move(board: Board, term: int) -> bool:
    if int(term) in {
        TERMINAL_ONGOING,
        TERMINAL_MAX_PLIES_DRAW,
        TERMINAL_REPETITION_DRAW,
        TERMINAL_NO_CAPTURE_DRAW,
    }:
        return False
    red_result = int(board.terminal_result_red_view(int(term)))
    if red_result == 0:
        return False
    stm_is_red = (int(board.turn()) == 0)
    return (red_result > 0) == stm_is_red


def _terminal_is_win_for_previous_mover(board: Board, term: int) -> bool:
    if int(term) in {
        TERMINAL_ONGOING,
        TERMINAL_MAX_PLIES_DRAW,
        TERMINAL_REPETITION_DRAW,
        TERMINAL_NO_CAPTURE_DRAW,
    }:
        return False
    red_result = int(board.terminal_result_red_view(int(term)))
    if red_result == 0:
        return False
    stm_is_red = (int(board.turn()) == 0)
    return (red_result > 0) != stm_is_red


def _side_to_move_has_check_forced_mate2(
    board: Board,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    for first in board.legal_moves():
        board.push(int(first))
        try:
            term_after_first = int(board.terminal_code(
                int(max_plies),
                int(repeat_limit),
                int(repeat_min_ply),
                int(no_capture_limit),
            ))
            if _terminal_is_win_for_previous_mover(board, term_after_first):
                return True
            if term_after_first != TERMINAL_ONGOING:
                continue
            if not bool(board.in_check_turn()):
                continue

            replies = list(board.legal_moves())
            if not replies:
                continue

            forced = True
            for reply in replies:
                board.push(int(reply))
                try:
                    term_after_reply = int(board.terminal_code(
                        int(max_plies),
                        int(repeat_limit),
                        int(repeat_min_ply),
                        int(no_capture_limit),
                    ))
                    if term_after_reply != TERMINAL_ONGOING:
                        reply_ok = _terminal_is_win_for_side_to_move(board, term_after_reply)
                    else:
                        reply_ok = _side_to_move_has_mate1(
                            board,
                            max_plies=max_plies,
                            repeat_limit=repeat_limit,
                            repeat_min_ply=repeat_min_ply,
                            no_capture_limit=no_capture_limit,
                        )
                finally:
                    board.pop()
                if not reply_ok:
                    forced = False
                    break
            if forced:
                return True
        finally:
            board.pop()
    return False


def _side_to_move_has_forcing_check_win(
    board: Board,
    *,
    plies_remaining: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    if plies_remaining < 1:
        return False

    legal = list(board.legal_moves())
    for first in legal:
        board.push(int(first))
        try:
            term_after_first = int(board.terminal_code(
                int(max_plies),
                int(repeat_limit),
                int(repeat_min_ply),
                int(no_capture_limit),
            ))
            if _terminal_is_win_for_previous_mover(board, term_after_first):
                return True
            if term_after_first != TERMINAL_ONGOING:
                continue
            if plies_remaining < 3:
                continue
            # Keep the defensive probe narrow: attacker continuations must be
            # checks. This targets forcing king-safety refutations rather than
            # broad strategic search.
            if not bool(board.in_check_turn()):
                continue

            replies = list(board.legal_moves())
            if not replies:
                continue

            forced = True
            for reply in replies:
                board.push(int(reply))
                try:
                    term_after_reply = int(board.terminal_code(
                        int(max_plies),
                        int(repeat_limit),
                        int(repeat_min_ply),
                        int(no_capture_limit),
                    ))
                    if term_after_reply != TERMINAL_ONGOING:
                        reply_ok = _terminal_is_win_for_side_to_move(board, term_after_reply)
                    else:
                        reply_ok = _side_to_move_has_forcing_check_win(
                            board,
                            plies_remaining=plies_remaining - 2,
                            max_plies=max_plies,
                            repeat_limit=repeat_limit,
                            repeat_min_ply=repeat_min_ply,
                            no_capture_limit=no_capture_limit,
                        )
                finally:
                    board.pop()
                if not reply_ok:
                    forced = False
                    break
            if forced:
                return True
        finally:
            board.pop()
    return False


def _move_allows_opponent_check_forced_mate2(
    board: Board,
    move: int,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    board.push(int(move))
    try:
        term = int(board.terminal_code(
            int(max_plies),
            int(repeat_limit),
            int(repeat_min_ply),
            int(no_capture_limit),
        ))
        if term != TERMINAL_ONGOING:
            return False
        return _side_to_move_has_check_forced_mate2(
            board,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
    finally:
        board.pop()


def _move_allows_opponent_forcing_check_win(
    board: Board,
    move: int,
    *,
    plies_remaining: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> bool:
    board.push(int(move))
    try:
        term = int(board.terminal_code(
            int(max_plies),
            int(repeat_limit),
            int(repeat_min_ply),
            int(no_capture_limit),
        ))
        if term != TERMINAL_ONGOING:
            return False
        return _side_to_move_has_forcing_check_win(
            board,
            plies_remaining=int(plies_remaining),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
    finally:
        board.pop()


def _apply_root_mate1_blunder_guard(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None]:
    """Refuse only moves that give the opponent an immediate terminal win.

    This is deliberately narrow: it does not add heuristic material rules or
    second-guess MCTS unless the selected root move walks into mate-in-1 and a
    non-losing root candidate exists.
    """

    if int(best_move) < 0 or not _move_allows_opponent_mate1(
        board,
        int(best_move),
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    ):
        return int(best_move), None

    # mcts_search returns root policy indices in the canonical policy frame when
    # canonical_policy=True.  Convert them back to raw internal moves before any
    # board legality/tactical checks. canonical_action is an involution for black.
    stm_is_black = bool(int(board.turn()) == 1)
    candidates: list[tuple[float, int]] = []
    for mv, prob in zip(list(idxs), list(probs)):
        raw_mv = int(canonical_action(int(mv), stm_is_black))
        candidates.append((float(prob), raw_mv))
    prob_by_move = {int(mv): float(prob) for prob, mv in candidates}
    candidates.sort(reverse=True)
    for _prob, mv in candidates:
        if mv == int(best_move):
            continue
        if not _move_allows_opponent_mate1(
            board,
            mv,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ):
            return int(mv), {
                "guard_type": "root_mate1_blunder_guard",
                "reason": "selected_move_allows_opponent_mate1",
                "original_move_uci": internal_move_to_uci(int(best_move)),
                "replacement_move_uci": internal_move_to_uci(int(mv)),
                "original_prob": prob_by_move.get(int(best_move)),
                "replacement_prob": prob_by_move.get(int(mv)),
            }
    return int(best_move), None


def _root_policy_candidates(idxs, probs, *, stm_is_black: bool) -> list[tuple[float, int]]:
    candidates: list[tuple[float, int]] = []
    for mv, prob in zip(list(idxs), list(probs)):
        raw_mv = int(canonical_action(int(mv), stm_is_black))
        candidates.append((float(prob), raw_mv))
    candidates.sort(reverse=True)
    return candidates


def _apply_root_gumbel_visit_selection(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    top_k: int,
    scale: float,
    seed: int,
) -> tuple[int, dict[str, Any] | None]:
    """Select from root visit candidates with a Gumbel perturbation.

    This is intentionally a lightweight root-level probe, not a full
    Gumbel-MuZero sequential-halving implementation inside the C++ MCTS.
    """

    if int(top_k) <= 1 or float(scale) <= 0.0:
        return int(best_move), None
    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    if len(candidates) <= 1:
        return int(best_move), None

    clipped = candidates[: max(1, int(top_k))]
    rng = random.Random(int(seed))
    scored: list[tuple[float, float, int]] = []
    for prob, mv in clipped:
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        gumbel = -math.log(-math.log(u))
        score = math.log(max(float(prob), 1e-12)) + float(scale) * gumbel
        scored.append((float(score), float(prob), int(mv)))
    scored.sort(reverse=True)
    _score, replacement_prob, replacement_move = scored[0]
    if int(replacement_move) == int(best_move):
        return int(best_move), None

    prob_by_move = {int(mv): float(prob) for prob, mv in candidates}
    return int(replacement_move), {
        "guard_type": "root_gumbel_visit_selection",
        "reason": "gumbelized_root_visit_candidate_selected",
        "original_move_uci": internal_move_to_uci(int(best_move)),
        "replacement_move_uci": internal_move_to_uci(int(replacement_move)),
        "original_prob": prob_by_move.get(int(best_move)),
        "replacement_prob": float(replacement_prob),
        "gumbel_top_k": int(top_k),
        "gumbel_scale": float(scale),
        "candidate_count": int(len(candidates)),
    }


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
        # After mover's candidate, the child side-to-move is the opponent.
        # Positive cp favors the child side-to-move, so a mover win is terrible
        # for the opponent and a mover loss is great for the opponent.
        return -20000 if mover_won else 20000
    finally:
        board.pop()


def _evaluate_child_for_opponent_cp(
    board: Board,
    move: int,
    verifier_pf: PikafishOpponent,
    *,
    depth: int,
    nodes: int,
    movetime_ms: int,
    max_wait_s: float,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, str, int | None]:
    terminal_cp = _terminal_child_eval_for_opponent(
        board,
        int(move),
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    )
    if terminal_cp is not None:
        return int(terminal_cp), "terminal", None

    board.push(int(move))
    try:
        child_fen = _pad_fen(board.fen())
    finally:
        board.pop()

    verifier_pf.set_position(child_fen)
    if int(depth) > 0:
        _best, _ponder, score_cp, mate_in = verifier_pf.go_depth_eval(
            int(depth),
            max_wait_s=float(max_wait_s),
        )
        mode = f"depth{int(depth)}"
    elif int(nodes) > 0:
        _best, _ponder, score_cp, mate_in = verifier_pf.go_nodes_eval(int(nodes), max_wait_s=120.0)
        mode = f"nodes{int(nodes)}"
    elif int(movetime_ms) > 0:
        _best, _ponder, score_cp, mate_in = verifier_pf.go_movetime_eval(
            int(movetime_ms),
            max_wait_s=float(movetime_ms) / 1000.0 + 60.0,
        )
        mode = f"movetime{int(movetime_ms)}"
    else:
        _best, _ponder, score_cp, mate_in = verifier_pf.go_depth_eval(1, max_wait_s=60.0)
        mode = "depth1"
    return int(score_cp), mode, mate_in


def _apply_root_pikafish_verifier(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    verifier_pf: PikafishOpponent,
    top_k: int,
    margin_cp: int,
    depth: int,
    nodes: int,
    movetime_ms: int,
    max_wait_s: float,
    danger_threshold_cp: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None]:
    if int(best_move) < 0:
        return int(best_move), None
    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    candidates = candidates[: max(1, int(top_k))]
    if all(int(move) != int(best_move) for _prob, move in candidates):
        candidates.insert(0, (1.0, int(best_move)))

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    original_row: dict[str, Any] | None = None
    for rank, (prob, move) in enumerate(candidates, start=1):
        move = int(move)
        if not bool(board.is_legal(move)):
            continue
        child_cp_opp, verifier_mode, mate_in = _evaluate_child_for_opponent_cp(
            board,
            move,
            verifier_pf,
            depth=depth,
            nodes=nodes,
            movetime_ms=movetime_ms,
            max_wait_s=float(max_wait_s),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        row = {
            "rank": int(rank),
            "move_uci": internal_move_to_uci(move),
            "prob": float(prob),
            "child_eval_cp_opponent_pov": int(child_cp_opp),
            "root_eval_cp_our_pov": int(-child_cp_opp),
            "mate_in": mate_in,
            "verifier_mode": verifier_mode,
        }
        rows.append(row)
        if move == int(best_move):
            original_row = row
        if best_row is None or int(child_cp_opp) < int(best_row["child_eval_cp_opponent_pov"]):
            best_row = row

    if best_row is None or original_row is None:
        return int(best_move), None
    original_cp = int(original_row["child_eval_cp_opponent_pov"])
    if int(danger_threshold_cp) > -20000 and original_cp < int(danger_threshold_cp):
        return int(best_move), None
    improvement_cp = original_cp - int(best_row["child_eval_cp_opponent_pov"])
    replacement_uci = str(best_row["move_uci"])
    if replacement_uci == internal_move_to_uci(int(best_move)) or improvement_cp < int(margin_cp):
        return int(best_move), None

    replacement_move = uci_move_to_internal(replacement_uci)
    return int(replacement_move), {
        "guard_type": "root_pikafish_topk_verifier",
        "reason": "verifier_found_safer_root_candidate",
        "original_move_uci": internal_move_to_uci(int(best_move)),
        "replacement_move_uci": replacement_uci,
        "original_child_eval_cp_opponent_pov": int(original_row["child_eval_cp_opponent_pov"]),
        "replacement_child_eval_cp_opponent_pov": int(best_row["child_eval_cp_opponent_pov"]),
        "improvement_cp": int(improvement_cp),
        "margin_cp": int(margin_cp),
        "danger_threshold_cp": int(danger_threshold_cp),
        "top_k": int(top_k),
        "candidates": rows,
    }


def _apply_root_mate2_blunder_guard(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None]:
    """Refuse moves that allow the opponent a check-forced mate-in-2."""

    if int(best_move) < 0 or not _move_allows_opponent_check_forced_mate2(
        board,
        int(best_move),
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    ):
        return int(best_move), None

    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    prob_by_move = {int(mv): float(prob) for prob, mv in candidates}
    for _prob, mv in candidates:
        if mv == int(best_move):
            continue
        if not _move_allows_opponent_check_forced_mate2(
            board,
            mv,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ):
            return int(mv), {
                "guard_type": "root_mate2_blunder_guard",
                "reason": "selected_move_allows_opponent_check_forced_mate2",
                "original_move_uci": internal_move_to_uci(int(best_move)),
                "replacement_move_uci": internal_move_to_uci(int(mv)),
                "original_prob": prob_by_move.get(int(best_move)),
                "replacement_prob": prob_by_move.get(int(mv)),
            }
    return int(best_move), None


def _apply_root_forcing_check_blunder_guard(
    board: Board,
    best_move: int,
    idxs,
    probs,
    *,
    plies_remaining: int,
    max_candidates: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None]:
    """Refuse moves that allow a bounded opponent check-forcing win."""

    if int(plies_remaining) < 3:
        return int(best_move), None
    if int(best_move) < 0 or not _move_allows_opponent_forcing_check_win(
        board,
        int(best_move),
        plies_remaining=int(plies_remaining),
        max_plies=max_plies,
        repeat_limit=repeat_limit,
        repeat_min_ply=repeat_min_ply,
        no_capture_limit=no_capture_limit,
    ):
        return int(best_move), None

    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    prob_by_move = {int(mv): float(prob) for prob, mv in candidates}
    checked = 0
    for _prob, mv in candidates:
        if mv == int(best_move):
            continue
        checked += 1
        if checked > int(max_candidates):
            break
        if not _move_allows_opponent_forcing_check_win(
            board,
            mv,
            plies_remaining=int(plies_remaining),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        ):
            return int(mv), {
                "guard_type": "root_forcing_check_blunder_guard",
                "reason": f"selected_move_allows_opponent_forcing_check_win_{int(plies_remaining)}ply",
                "original_move_uci": internal_move_to_uci(int(best_move)),
                "replacement_move_uci": internal_move_to_uci(int(mv)),
                "original_prob": prob_by_move.get(int(best_move)),
                "replacement_prob": prob_by_move.get(int(mv)),
                "forcing_plies": int(plies_remaining),
                "checked_alternatives": int(checked),
            }
    return int(best_move), None


def _run_shadow_value_probe(
    board: Board,
    evaluator,
    *,
    actual_move: int,
    value_source: str,
    num_simulations: int,
    c_puct: float,
    c_puct_base: float,
    c_puct_factor: float,
    temperature_move: float,
    q_weight: float,
    q_clip: float,
    fpu_reduction_root: float,
    fpu_reduction_tree: float,
    seed: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
    tactical_mate1_extension: bool,
    tactical_mate2_extension: bool,
    top_k: int,
) -> dict[str, Any]:
    best_move, idxs, probs, root_v = mcts_search(
        board=board,
        net=evaluator,
        num_simulations=int(num_simulations),
        c_puct=float(c_puct),
        q_weight=float(q_weight),
        q_clip=float(q_clip),
        add_root_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
        temperature_move=float(temperature_move),
        temperature_target=1.0,
        eval_batch_size=16,
        seed=int(seed),
        canonical_input=True,
        canonical_policy=True,
        max_plies=int(max_plies),
        repeat_limit=int(repeat_limit),
        repeat_min_ply=int(repeat_min_ply),
        no_capture_limit=int(no_capture_limit),
        tactical_mate1_extension=bool(tactical_mate1_extension),
        tactical_mate2_extension=bool(tactical_mate2_extension),
        c_puct_base=float(c_puct_base),
        c_puct_factor=float(c_puct_factor),
        fpu_reduction_root=float(fpu_reduction_root),
        fpu_reduction_tree=float(fpu_reduction_tree),
    )
    candidates = _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))
    top_rows = [
        {
            "rank": int(rank),
            "move_raw": int(move),
            "move_uci": internal_move_to_uci(int(move)),
            "visit_prob": float(prob),
        }
        for rank, (prob, move) in enumerate(candidates[: max(1, int(top_k))], start=1)
    ]
    best_uci = internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else ""
    actual_uci = internal_move_to_uci(int(actual_move)) if int(actual_move) >= 0 else ""
    actual_rank = None
    actual_visit_prob = None
    for row in top_rows:
        if row["move_uci"] == actual_uci:
            actual_rank = int(row["rank"])
            actual_visit_prob = float(row["visit_prob"])
            break
    return {
        "shadow_value_source": str(value_source),
        "shadow_sims": int(num_simulations),
        "shadow_root_value": float(root_v),
        "shadow_best_move_uci": best_uci,
        "actual_move_uci": actual_uci,
        "disagrees_with_actual": bool(best_uci and actual_uci and best_uci != actual_uci),
        "actual_rank_in_shadow_topk": actual_rank,
        "actual_visit_prob_in_shadow_topk": actual_visit_prob,
        "shadow_top_moves": top_rows,
    }


def _apply_root_shadow_disagreement_verifier(
    board: Board,
    best_move: int,
    idxs,
    probs,
    shadow_probe: dict[str, Any],
    *,
    verifier_pf: PikafishOpponent,
    top_k: int,
    margin_cp: int,
    ordinary_min_original_cp: int,
    mate_risk_margin_cp: int,
    mate_risk_cp: int,
    mate_risk_safe_cp: int,
    escape_margin_cp: int,
    escape_risk_cp: int,
    escape_safe_cp: int,
    trigger_rank: int,
    trigger_min_gap: float,
    ambiguous_gap: float,
    depth: int,
    nodes: int,
    movetime_ms: int,
    max_wait_s: float,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any]]:
    attempt: dict[str, Any] = {
        "enabled": True,
        "attempted": False,
        "accepted": False,
        "reason": "",
        "margin_cp": int(margin_cp),
        "ordinary_min_original_cp": int(ordinary_min_original_cp),
        "mate_risk_margin_cp": int(mate_risk_margin_cp),
        "mate_risk_cp": int(mate_risk_cp),
        "mate_risk_safe_cp": int(mate_risk_safe_cp),
        "escape_margin_cp": int(escape_margin_cp),
        "escape_risk_cp": int(escape_risk_cp),
        "escape_safe_cp": int(escape_safe_cp),
        "trigger_rank": int(trigger_rank),
        "trigger_min_gap": float(trigger_min_gap),
        "ambiguous_gap": float(ambiguous_gap),
        "top_k": int(top_k),
    }
    if int(best_move) < 0:
        attempt["reason"] = "invalid_original_move"
        return int(best_move), None, attempt

    shadow_top = list(shadow_probe.get("shadow_top_moves") or [])
    actual_rank = shadow_probe.get("actual_rank_in_shadow_topk")
    actual_prob = shadow_probe.get("actual_visit_prob_in_shadow_topk")
    top_prob = float(shadow_top[0].get("visit_prob", 0.0) or 0.0) if shadow_top else 0.0
    second_prob = float(shadow_top[1].get("visit_prob", 0.0) or 0.0) if len(shadow_top) > 1 else 0.0
    actual_prob_f = float(actual_prob) if actual_prob is not None else None
    rank_trigger = False
    rank_gap = None
    if int(trigger_rank) > 0:
        rank_bad = actual_rank is None or int(actual_rank) > int(trigger_rank)
        if actual_prob_f is None:
            rank_gap = None
            gap_ok = True
        else:
            rank_gap = float(top_prob - actual_prob_f)
            gap_ok = rank_gap >= float(trigger_min_gap)
        rank_trigger = bool(rank_bad and gap_ok)
    ambiguous_trigger = False
    ambiguous_top_gap = None
    if float(ambiguous_gap) >= 0.0 and len(shadow_top) > 1:
        ambiguous_top_gap = float(top_prob - second_prob)
        ambiguous_trigger = bool(ambiguous_top_gap <= float(ambiguous_gap))

    disagreement_trigger = bool(shadow_probe.get("disagrees_with_actual"))
    if disagreement_trigger:
        trigger_reason = "scalar_shadow_disagreement"
    elif rank_trigger:
        trigger_reason = "shadow_rank_trigger"
    elif ambiguous_trigger:
        trigger_reason = "shadow_ambiguous_top_trigger"
    else:
        trigger_reason = "no_scalar_shadow_disagreement"
    attempt.update(
        {
            "trigger_reason": trigger_reason,
            "actual_rank_in_shadow_topk": actual_rank,
            "actual_visit_prob_in_shadow_topk": actual_prob,
            "shadow_top1_visit_prob": top_prob,
            "shadow_top2_visit_prob": second_prob if len(shadow_top) > 1 else None,
            "shadow_rank_gap": rank_gap,
            "shadow_ambiguous_top_gap": ambiguous_top_gap,
        }
    )
    if not bool(disagreement_trigger or rank_trigger or ambiguous_trigger):
        attempt["reason"] = "no_scalar_shadow_disagreement"
        return int(best_move), None, attempt

    candidates: dict[int, dict[str, Any]] = {}

    def add_candidate(move: int, *, source: str, rank: int | None = None, prob: float | None = None) -> None:
        move = int(move)
        if not bool(board.is_legal(move)):
            return
        row = candidates.setdefault(
            move,
            {
                "move": move,
                "move_uci": internal_move_to_uci(move),
                "sources": [],
                "ranks": {},
                "probs": {},
            },
        )
        if source not in row["sources"]:
            row["sources"].append(source)
        if rank is not None:
            row["ranks"][source] = int(rank)
        if prob is not None:
            row["probs"][source] = float(prob)

    add_candidate(int(best_move), source="actual", rank=1, prob=None)
    for rank, (prob, move) in enumerate(
        _root_policy_candidates(idxs, probs, stm_is_black=bool(int(board.turn()) == 1))[: max(1, int(top_k))],
        start=1,
    ):
        add_candidate(int(move), source="scalar_topk", rank=int(rank), prob=float(prob))
    for row in (shadow_probe.get("shadow_top_moves") or [])[: max(1, int(top_k))]:
        raw = row.get("move_raw")
        if raw is None:
            try:
                raw = int(uci_move_to_internal(str(row.get("move_uci", ""))[:4]))
            except Exception:
                continue
        add_candidate(
            int(raw),
            source="shadow_topk",
            rank=int(row.get("rank", 999)),
            prob=float(row.get("visit_prob", 0.0) or 0.0),
        )

    attempt["attempted"] = True
    attempt["candidate_count"] = int(len(candidates))
    attempt["original_move_uci"] = internal_move_to_uci(int(best_move))
    attempt["shadow_best_move_uci"] = str(shadow_probe.get("shadow_best_move_uci", ""))
    attempt["shadow_value_source"] = str(shadow_probe.get("shadow_value_source", ""))

    rows: list[dict[str, Any]] = []
    original_row: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None
    for cand in candidates.values():
        move = int(cand["move"])
        child_cp_opp, verifier_mode, mate_in = _evaluate_child_for_opponent_cp(
            board,
            move,
            verifier_pf,
            depth=int(depth),
            nodes=int(nodes),
            movetime_ms=int(movetime_ms),
            max_wait_s=float(max_wait_s),
            max_plies=int(max_plies),
            repeat_limit=int(repeat_limit),
            repeat_min_ply=int(repeat_min_ply),
            no_capture_limit=int(no_capture_limit),
        )
        row = {
            "move_uci": str(cand["move_uci"]),
            "sources": list(cand["sources"]),
            "ranks": dict(cand["ranks"]),
            "probs": dict(cand["probs"]),
            "child_eval_cp_opponent_pov": int(child_cp_opp),
            "root_eval_cp_our_pov": int(-child_cp_opp),
            "mate_in": mate_in,
            "verifier_mode": verifier_mode,
        }
        rows.append(row)
        if move == int(best_move):
            original_row = row
        if best_row is None or int(child_cp_opp) < int(best_row["child_eval_cp_opponent_pov"]):
            best_row = row

    if original_row is None or best_row is None:
        attempt["reason"] = "missing_original_or_best_candidate"
        attempt["candidates"] = rows
        return int(best_move), None, attempt
    original_cp = int(original_row["child_eval_cp_opponent_pov"])
    best_cp = int(best_row["child_eval_cp_opponent_pov"])
    improvement_cp = original_cp - best_cp
    replacement_uci = str(best_row["move_uci"])
    mate_risk_enabled = int(mate_risk_margin_cp) >= 0
    original_mate_risk = (
        original_row.get("mate_in") is not None
        or original_cp >= int(mate_risk_cp)
    )
    replacement_mate_risk = (
        best_row.get("mate_in") is not None
        or best_cp >= int(mate_risk_cp)
    )
    mate_risk_safe_cap_enabled = int(mate_risk_safe_cp) < int(mate_risk_cp)
    replacement_safe_for_mate_risk = (
        not bool(mate_risk_safe_cap_enabled)
        or best_cp <= int(mate_risk_safe_cp)
    )
    escape_enabled = int(escape_margin_cp) >= 0
    escape_risk_original = bool(escape_enabled and original_cp >= int(escape_risk_cp))
    replacement_safe_for_escape = (
        not bool(escape_risk_original)
        or best_cp <= int(escape_safe_cp)
    )
    ordinary_accept = (
        improvement_cp >= int(margin_cp)
        and original_cp >= int(ordinary_min_original_cp)
    )
    if bool(original_mate_risk) and not bool(replacement_safe_for_mate_risk):
        ordinary_accept = False
    if bool(escape_risk_original) and not bool(replacement_safe_for_escape):
        ordinary_accept = False
    mate_risk_accept = (
        mate_risk_enabled
        and bool(original_mate_risk)
        and not bool(replacement_mate_risk)
        and bool(replacement_safe_for_mate_risk)
        and improvement_cp >= int(mate_risk_margin_cp)
    )
    escape_accept = (
        escape_enabled
        and original_cp >= int(escape_risk_cp)
        and best_cp <= int(escape_safe_cp)
        and improvement_cp >= int(escape_margin_cp)
    )
    attempt.update(
        {
            "reason": "accepted",
            "best_move_uci": replacement_uci,
            "original_child_eval_cp_opponent_pov": original_cp,
            "best_child_eval_cp_opponent_pov": best_cp,
            "improvement_cp": int(improvement_cp),
            "original_mate_risk": bool(original_mate_risk),
            "replacement_mate_risk": bool(replacement_mate_risk),
            "replacement_safe_for_mate_risk": bool(replacement_safe_for_mate_risk),
            "escape_risk_original": bool(escape_risk_original),
            "replacement_safe_for_escape": bool(replacement_safe_for_escape),
            "ordinary_accept": bool(ordinary_accept),
            "ordinary_min_original_cp": int(ordinary_min_original_cp),
            "mate_risk_accept": bool(mate_risk_accept),
            "escape_accept": bool(escape_accept),
            "candidates": rows,
        }
    )
    if replacement_uci == internal_move_to_uci(int(best_move)):
        attempt["reason"] = "verified_original_best"
        return int(best_move), None, attempt
    if not bool(ordinary_accept or mate_risk_accept or escape_accept):
        attempt["reason"] = "improvement_below_margin"
        return int(best_move), None, attempt

    replacement_move = int(uci_move_to_internal(replacement_uci))
    attempt["accepted"] = True
    if bool(mate_risk_accept and not ordinary_accept):
        acceptance_rule = "mate_risk"
    elif bool(escape_accept and not ordinary_accept):
        acceptance_rule = "escape"
    else:
        acceptance_rule = "ordinary"
    return replacement_move, {
        "guard_type": "shadow_value_disagreement_verifier",
        "reason": (
            "scalar_wdl_disagreement_verified_mate_risk_candidate"
            if acceptance_rule == "mate_risk"
            else "scalar_wdl_disagreement_verified_escape_candidate"
            if acceptance_rule == "escape"
            else "scalar_wdl_disagreement_verified_safer_candidate"
        ),
        "original_move_uci": internal_move_to_uci(int(best_move)),
        "replacement_move_uci": replacement_uci,
        "shadow_best_move_uci": str(shadow_probe.get("shadow_best_move_uci", "")),
        "shadow_value_source": str(shadow_probe.get("shadow_value_source", "")),
        "trigger_reason": trigger_reason,
        "original_child_eval_cp_opponent_pov": original_cp,
        "replacement_child_eval_cp_opponent_pov": best_cp,
        "improvement_cp": int(improvement_cp),
        "margin_cp": int(margin_cp),
        "ordinary_min_original_cp": int(ordinary_min_original_cp),
        "mate_risk_margin_cp": int(mate_risk_margin_cp),
        "mate_risk_cp": int(mate_risk_cp),
        "mate_risk_safe_cp": int(mate_risk_safe_cp),
        "escape_margin_cp": int(escape_margin_cp),
        "escape_risk_cp": int(escape_risk_cp),
        "escape_safe_cp": int(escape_safe_cp),
        "shadow_trigger_rank": int(trigger_rank),
        "shadow_trigger_min_gap": float(trigger_min_gap),
        "shadow_ambiguous_gap": float(ambiguous_gap),
        "acceptance_rule": acceptance_rule,
        "original_mate_risk": bool(original_mate_risk),
        "replacement_mate_risk": bool(replacement_mate_risk),
        "replacement_safe_for_mate_risk": bool(replacement_safe_for_mate_risk),
        "escape_risk_original": bool(escape_risk_original),
        "replacement_safe_for_escape": bool(replacement_safe_for_escape),
        "top_k": int(top_k),
        "candidates": rows,
    }, attempt


def _our_pick_move(
    board: Board,
    evaluator,
    *,
    search_kind: str,
    num_simulations: int,
    c_puct: float,
    c_puct_base: float,
    c_puct_factor: float,
    temperature_move: float,
    q_weight: float,
    q_clip: float,
    fpu_reduction_root: float,
    fpu_reduction_tree: float,
    add_root_noise: bool,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    seed: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
    ab_depth: int,
    ab_root_max_branch: int,
    ab_max_branch: int,
    ab_quiescence_depth: int,
    ab_quiescence_max_branch: int,
    ab_tt_mb: int,
    tactical_mate1_extension: bool,
    tactical_mate2_extension: bool,
    root_mate1_blunder_guard: bool,
    root_mate2_blunder_guard: bool,
    root_forcing_check_guard_plies: int,
    root_forcing_check_guard_max_candidates: int,
    root_selection_mode: str,
    root_gumbel_top_k: int,
    root_gumbel_scale: float,
    log_root_stats_top_k: int,
    verifier_pf: PikafishOpponent | None,
    verifier_top_k: int,
    verifier_margin_cp: int,
    verifier_depth: int,
    verifier_nodes: int,
    verifier_movetime_ms: int,
    verifier_max_wait_s: float,
    verifier_danger_threshold_cp: int,
    shadow_verifier_pf: PikafishOpponent | None,
    danger_runtime: _DangerHeadRuntime | None,
    danger_mode: str,
    danger_top_k: int,
    danger_lambda: float,
    danger_veto_threshold: float,
    danger_triage_threshold: float,
    danger_triage_exact_plies: int,
    danger_triage_exact_max_candidates: int,
    shadow_evaluator=None,
    shadow_value_source: str = "none",
    shadow_sims: int = 0,
    shadow_top_k: int = 8,
    shadow_disagreement_verifier: bool = False,
    shadow_verifier_top_k: int = 6,
    shadow_verifier_margin_cp: int = 300,
    shadow_verifier_ordinary_min_original_cp: int = -20000,
    shadow_verifier_mate_risk_margin_cp: int = -1,
    shadow_verifier_mate_risk_cp: int = 19000,
    shadow_verifier_mate_risk_safe_cp: int = 19000,
    shadow_verifier_escape_margin_cp: int = -1,
    shadow_verifier_escape_risk_cp: int = 500,
    shadow_verifier_escape_safe_cp: int = 100,
    shadow_verifier_trigger_rank: int = 0,
    shadow_verifier_trigger_min_gap: float = 0.0,
    shadow_verifier_ambiguous_gap: float = -1.0,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    search_kind = str(search_kind)
    search_info: dict[str, Any] | None = None
    if search_kind == "alphabeta":
        ab_config = AlphaBetaConfig(
            depth=int(ab_depth),
            root_max_branch=int(ab_root_max_branch),
            max_branch=int(ab_max_branch),
            quiescence_depth=int(ab_quiescence_depth),
            quiescence_max_branch=int(ab_quiescence_max_branch),
            tt_mb=int(ab_tt_mb),
            max_plies=int(max_plies),
            repeat_limit=int(repeat_limit),
            repeat_min_ply=int(repeat_min_ply),
            no_capture_limit=int(no_capture_limit),
        )
        best_move, idxs, probs, root_v, ab_stats = alpha_beta_search(
            board=board,
            evaluator=evaluator,
            config=ab_config,
        )
        search_info = {
            "search_kind": "alphabeta",
            "root_value": float(root_v),
            "stats": ab_stats,
        }
    else:
        mcts_kwargs = dict(
            board=board,
            net=evaluator,
            num_simulations=int(num_simulations),
            c_puct=float(c_puct),
            q_weight=float(q_weight),
            q_clip=float(q_clip),
            add_root_noise=bool(add_root_noise),
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_eps=float(dirichlet_eps),
            temperature_move=float(temperature_move),
            temperature_target=1.0,
            eval_batch_size=16,
            seed=int(seed),
            canonical_input=True,
            canonical_policy=True,
            max_plies=int(max_plies),
            repeat_limit=int(repeat_limit),
            repeat_min_ply=int(repeat_min_ply),
            no_capture_limit=int(no_capture_limit),
            tactical_mate1_extension=bool(tactical_mate1_extension),
            tactical_mate2_extension=bool(tactical_mate2_extension),
            c_puct_base=float(c_puct_base),
            c_puct_factor=float(c_puct_factor),
            fpu_reduction_root=float(fpu_reduction_root),
            fpu_reduction_tree=float(fpu_reduction_tree),
        )
        root_stats = []
        if int(log_root_stats_top_k) > 0:
            try:
                best_move, idxs, probs, root_v, root_stats = mcts_search_with_root_stats(**mcts_kwargs)
            except AttributeError:
                best_move, idxs, probs, root_v = mcts_search(**mcts_kwargs)
                root_stats = []
        else:
            best_move, idxs, probs, root_v = mcts_search(**mcts_kwargs)
        search_info = {
            "search_kind": "mcts",
            "root_value": float(root_v),
            "root_best_move_uci": internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else "",
        }
        if int(log_root_stats_top_k) > 0:
            stats_rows = []
            sorted_root_stats = sorted(
                [dict(row) for row in list(root_stats)],
                key=lambda row: float(row.get("visit_prob", 0.0)),
                reverse=True,
            )
            for rank, raw in enumerate(sorted_root_stats, start=1):
                row = dict(raw)
                move = int(row.get("move_raw", -1))
                if move < 0:
                    continue
                stats_rows.append(
                    {
                        "move_uci": internal_move_to_uci(move),
                        "move_raw": move,
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
                )
                if len(stats_rows) >= int(log_root_stats_top_k):
                    break
            search_info["root_stats"] = stats_rows
            search_info["root_stats_top_k"] = int(log_root_stats_top_k)
    guard_event = None
    if str(root_selection_mode) == "gumbel_visit":
        best_move, gumbel_event = _apply_root_gumbel_visit_selection(
            board,
            int(best_move),
            idxs,
            probs,
            top_k=int(root_gumbel_top_k),
            scale=float(root_gumbel_scale),
            seed=int(seed) ^ 0x5F3759DF,
        )
        if gumbel_event is not None:
            guard_event = gumbel_event
    if verifier_pf is not None and int(verifier_top_k) > 0:
        best_move, verifier_event = _apply_root_pikafish_verifier(
            board,
            int(best_move),
            idxs,
            probs,
            verifier_pf=verifier_pf,
            top_k=int(verifier_top_k),
            margin_cp=int(verifier_margin_cp),
            depth=int(verifier_depth),
            nodes=int(verifier_nodes),
            movetime_ms=int(verifier_movetime_ms),
            max_wait_s=float(verifier_max_wait_s),
            danger_threshold_cp=int(verifier_danger_threshold_cp),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if verifier_event is not None:
            if guard_event is not None:
                verifier_event["prior_guard_event"] = guard_event
            guard_event = verifier_event
    if danger_runtime is not None and str(danger_mode) == "rerank":
        best_move, danger_event = _apply_root_danger_head_rerank(
            board,
            int(best_move),
            idxs,
            probs,
            danger_runtime=danger_runtime,
            top_k=int(danger_top_k),
            danger_lambda=float(danger_lambda),
            veto_threshold=float(danger_veto_threshold),
        )
        if danger_event is not None:
            guard_event = danger_event
    elif danger_runtime is not None and str(danger_mode) == "triage":
        best_move, danger_event = _apply_root_danger_head_triage(
            board,
            int(best_move),
            idxs,
            probs,
            danger_runtime=danger_runtime,
            top_k=int(danger_top_k),
            triage_threshold=float(danger_triage_threshold),
            exact_plies=int(danger_triage_exact_plies),
            exact_max_candidates=int(danger_triage_exact_max_candidates),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if danger_event is not None:
            guard_event = danger_event
    if shadow_evaluator is not None and str(shadow_value_source) != "none" and search_kind == "mcts":
        sims = int(shadow_sims) if int(shadow_sims) > 0 else int(num_simulations)
        shadow = _run_shadow_value_probe(
            board,
            shadow_evaluator,
            actual_move=int(best_move),
            value_source=str(shadow_value_source),
            num_simulations=sims,
            c_puct=float(c_puct),
            c_puct_base=float(c_puct_base),
            c_puct_factor=float(c_puct_factor),
            temperature_move=float(temperature_move),
            q_weight=float(q_weight),
            q_clip=float(q_clip),
            fpu_reduction_root=float(fpu_reduction_root),
            fpu_reduction_tree=float(fpu_reduction_tree),
            seed=int(seed) ^ 0x51A7E,
            max_plies=int(max_plies),
            repeat_limit=int(repeat_limit),
            repeat_min_ply=int(repeat_min_ply),
            no_capture_limit=int(no_capture_limit),
            tactical_mate1_extension=bool(tactical_mate1_extension),
            tactical_mate2_extension=bool(tactical_mate2_extension),
            top_k=int(shadow_top_k),
        )
        if search_info is None:
            search_info = {"search_kind": "mcts"}
        search_info["shadow_value_probe"] = shadow
        if bool(shadow_disagreement_verifier) and shadow_verifier_pf is not None:
            best_move, shadow_event, shadow_attempt = _apply_root_shadow_disagreement_verifier(
                board,
                int(best_move),
                idxs,
                probs,
                shadow,
                verifier_pf=shadow_verifier_pf,
                top_k=int(shadow_verifier_top_k),
                margin_cp=int(shadow_verifier_margin_cp),
                ordinary_min_original_cp=int(shadow_verifier_ordinary_min_original_cp),
                mate_risk_margin_cp=int(shadow_verifier_mate_risk_margin_cp),
                mate_risk_cp=int(shadow_verifier_mate_risk_cp),
                mate_risk_safe_cp=int(shadow_verifier_mate_risk_safe_cp),
                escape_margin_cp=int(shadow_verifier_escape_margin_cp),
                escape_risk_cp=int(shadow_verifier_escape_risk_cp),
                escape_safe_cp=int(shadow_verifier_escape_safe_cp),
                trigger_rank=int(shadow_verifier_trigger_rank),
                trigger_min_gap=float(shadow_verifier_trigger_min_gap),
                ambiguous_gap=float(shadow_verifier_ambiguous_gap),
                depth=int(verifier_depth),
                nodes=int(verifier_nodes),
                movetime_ms=int(verifier_movetime_ms),
                max_wait_s=float(verifier_max_wait_s),
                max_plies=int(max_plies),
                repeat_limit=int(repeat_limit),
                repeat_min_ply=int(repeat_min_ply),
                no_capture_limit=int(no_capture_limit),
            )
            shadow["shadow_disagreement_verifier_attempt"] = shadow_attempt
            shadow["move_after_shadow_gate_uci"] = internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else ""
            if shadow_event is not None:
                if guard_event is not None:
                    shadow_event["prior_guard_event"] = guard_event
                guard_event = shadow_event
    if root_mate1_blunder_guard:
        best_move, mate1_event = _apply_root_mate1_blunder_guard(
            board,
            int(best_move),
            idxs,
            probs,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if mate1_event is not None:
            if guard_event is not None:
                mate1_event["prior_guard_event"] = guard_event
            guard_event = mate1_event
    if root_mate2_blunder_guard:
        best_move, mate2_event = _apply_root_mate2_blunder_guard(
            board,
            int(best_move),
            idxs,
            probs,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if mate2_event is not None:
            if guard_event is not None:
                mate2_event["prior_guard_event"] = guard_event
            guard_event = mate2_event
    if int(root_forcing_check_guard_plies) >= 3:
        best_move, forcing_event = _apply_root_forcing_check_blunder_guard(
            board,
            int(best_move),
            idxs,
            probs,
            plies_remaining=int(root_forcing_check_guard_plies),
            max_candidates=int(root_forcing_check_guard_max_candidates),
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if forcing_event is not None:
            if guard_event is not None:
                forcing_event["prior_guard_event"] = guard_event
            guard_event = forcing_event
    if search_info is not None and isinstance(search_info.get("shadow_value_probe"), dict):
        search_info["shadow_value_probe"]["final_move_after_all_guards_uci"] = (
            internal_move_to_uci(int(best_move)) if int(best_move) >= 0 else ""
        )
    return int(best_move), guard_event, search_info


def _elo_from_score(score: float) -> float:
    if score <= 1e-4:
        return -2000.0
    if score >= 1.0 - 1e-4:
        return 2000.0
    return -400.0 * math.log10((1.0 - score) / score)


def _play_one_arena_game(
    *,
    gi: int,
    args,
    evaluator,
    pf: PikafishOpponent | None,
    verifier_pf: PikafishOpponent | None,
    rng: random.Random,
    opening_entry: dict[str, Any] | None = None,
    opening_index: int | None = None,
) -> tuple[GameRecord, int, int]:
    """Play one full game against the configured opponent.  Returns (record, term_code, plies).

    Pure per-game function — no shared state mutation.  Caller is responsible for
    threading-safe accumulation of GameRecords + counts.
    """
    side_mode = str(getattr(args, "our_side", "alternate"))
    if side_mode == "red":
        our_is_red = True
    elif side_mode == "black":
        our_is_red = False
    else:
        our_is_red = (gi % 2 == 0)
    our_side = "red" if our_is_red else "black"
    board = Board()
    _initialize_opening(board, opening_entry)
    rec = GameRecord(
        index=gi,
        our_side=our_side,
        opening_fen=board.fen(),
        opening_id=str(opening_entry.get("id", "")) if opening_entry is not None else "",
        opening_index=opening_index,
    )

    if pf is not None:
        pf.new_game()
    ply = 0
    game_result: str | None = None
    game_termination = -1

    while True:
        term = int(board.terminal_code(
            int(args.max_plies),
            int(args.repeat_limit),
            int(args.repeat_min_ply),
            int(args.no_capture_limit),
        ))
        if term != TERMINAL_ONGOING:
            game_termination = term
            red_result = int(board.terminal_result_red_view(term))
            if red_result == 0:
                game_result = "draw"
            elif (red_result > 0) == our_is_red:
                game_result = "our_win"
            else:
                game_result = "opp_win"
            break

        red_to_move = (int(board.turn()) == 0)
        our_turn = (red_to_move == our_is_red)

        if our_turn:
            fen_before = _pad_fen(board.fen())
            move, guard_event, search_info = _our_pick_move(
                board,
                evaluator,
                search_kind=str(args.our_search),
                num_simulations=args.our_sims,
                c_puct=args.our_c_puct,
                c_puct_base=args.our_c_puct_base,
                c_puct_factor=args.our_c_puct_factor,
                temperature_move=args.our_temperature_move,
                q_weight=args.our_q_weight,
                q_clip=args.our_q_clip,
                fpu_reduction_root=args.our_fpu_reduction_root,
                fpu_reduction_tree=args.our_fpu_reduction_tree,
                add_root_noise=bool(args.our_add_root_noise),
                dirichlet_alpha=args.our_dirichlet_alpha,
                dirichlet_eps=args.our_dirichlet_eps,
                seed=int(args.seed + gi * 10_007 + ply * 31),
                max_plies=args.max_plies,
                repeat_limit=args.repeat_limit,
                repeat_min_ply=args.repeat_min_ply,
                no_capture_limit=args.no_capture_limit,
                ab_depth=int(args.ab_depth),
                ab_root_max_branch=int(args.ab_root_max_branch),
                ab_max_branch=int(args.ab_max_branch),
                ab_quiescence_depth=int(args.ab_quiescence_depth),
                ab_quiescence_max_branch=int(args.ab_quiescence_max_branch),
                ab_tt_mb=int(args.ab_tt_mb),
                tactical_mate1_extension=bool(args.our_tactical_mate1_extension),
                tactical_mate2_extension=bool(args.our_tactical_mate2_extension),
                root_mate1_blunder_guard=bool(args.our_root_mate1_blunder_guard),
                root_mate2_blunder_guard=bool(args.our_root_mate2_blunder_guard),
                root_forcing_check_guard_plies=int(args.our_root_forcing_check_guard_plies),
                root_forcing_check_guard_max_candidates=int(args.our_root_forcing_check_guard_max_candidates),
                root_selection_mode=str(args.our_root_selection_mode),
                root_gumbel_top_k=int(args.our_root_gumbel_top_k),
                root_gumbel_scale=float(args.our_root_gumbel_scale),
                log_root_stats_top_k=int(args.our_log_root_stats_top_k),
                verifier_pf=verifier_pf if (
                    bool(args.our_pikafish_verifier)
                    and str(args.our_verifier_side) in ("any", our_side)
                ) else None,
                verifier_top_k=int(args.our_verifier_top_k),
                verifier_margin_cp=int(args.our_verifier_margin_cp),
                verifier_depth=int(args.our_verifier_depth),
                verifier_nodes=int(args.our_verifier_nodes),
                verifier_movetime_ms=int(args.our_verifier_movetime_ms),
                verifier_max_wait_s=float(args.our_verifier_max_wait_s),
                verifier_danger_threshold_cp=int(args.our_verifier_danger_threshold_cp),
                shadow_verifier_pf=verifier_pf if (
                    bool(args.our_shadow_disagreement_verifier)
                    and str(args.our_shadow_value_source) != "none"
                    and str(args.our_shadow_side) in ("any", our_side)
                ) else None,
                danger_runtime=getattr(args, "_danger_runtime", None),
                danger_mode=str(args.our_danger_mode),
                danger_top_k=int(args.our_danger_top_k),
                danger_lambda=float(args.our_danger_lambda),
                danger_veto_threshold=float(args.our_danger_veto_threshold),
                danger_triage_threshold=float(args.our_danger_triage_threshold),
                danger_triage_exact_plies=int(args.our_danger_triage_exact_plies),
                danger_triage_exact_max_candidates=int(args.our_danger_triage_exact_max_candidates),
                shadow_evaluator=getattr(args, "_shadow_evaluator", None) if (
                    str(args.our_shadow_value_source) != "none"
                    and str(args.our_shadow_side) in ("any", our_side)
                ) else None,
                shadow_value_source=str(args.our_shadow_value_source),
                shadow_sims=int(args.our_shadow_sims),
                shadow_top_k=int(args.our_shadow_top_k),
                shadow_disagreement_verifier=bool(args.our_shadow_disagreement_verifier),
                shadow_verifier_top_k=int(args.our_shadow_verifier_top_k),
                shadow_verifier_margin_cp=int(args.our_shadow_verifier_margin_cp),
                shadow_verifier_ordinary_min_original_cp=int(args.our_shadow_verifier_ordinary_min_original_cp),
                shadow_verifier_mate_risk_margin_cp=int(args.our_shadow_verifier_mate_risk_margin_cp),
                shadow_verifier_mate_risk_cp=int(args.our_shadow_verifier_mate_risk_cp),
                shadow_verifier_mate_risk_safe_cp=int(args.our_shadow_verifier_mate_risk_safe_cp),
                shadow_verifier_escape_margin_cp=int(args.our_shadow_verifier_escape_margin_cp),
                shadow_verifier_escape_risk_cp=int(args.our_shadow_verifier_escape_risk_cp),
                shadow_verifier_escape_safe_cp=int(args.our_shadow_verifier_escape_safe_cp),
                shadow_verifier_trigger_rank=int(args.our_shadow_verifier_trigger_rank),
                shadow_verifier_trigger_min_gap=float(args.our_shadow_verifier_trigger_min_gap),
                shadow_verifier_ambiguous_gap=float(args.our_shadow_verifier_ambiguous_gap),
            )
            if guard_event is not None:
                enriched_event = {
                    "game_index": int(gi),
                    "ply": int(ply),
                    "side": "red" if red_to_move else "black",
                    "fen_before": fen_before,
                }
                enriched_event.update(guard_event)
                rec.guard_events.append(enriched_event)
            if move < 0:
                game_termination = TERMINAL_CHECKMATE_OR_STALEMATE
                red_result = int(board.terminal_result_red_view(game_termination))
                game_result = "draw" if red_result == 0 else (
                    "our_win" if (red_result > 0) == our_is_red else "opp_win"
                )
                break
            uci = internal_move_to_uci(move)
            if search_info is not None and (
                str(search_info.get("search_kind", "")) != "mcts"
                or "shadow_value_probe" in search_info
                or "root_stats" in search_info
            ):
                search_row = {
                    "game_index": int(gi),
                    "ply": int(ply),
                    "side": "red" if red_to_move else "black",
                    "move_uci": uci,
                }
                search_row.update(search_info)
                rec.search_stats.append(search_row)
        else:
            if args.opp_random:
                legal = board.legal_moves()
                if not legal:
                    move = -1
                    uci = "0000"
                else:
                    move = int(rng.choice(list(legal)))
                    uci = internal_move_to_uci(move)
            elif (args.opp_noise_ratio > 0.0
                  and rng.random() < float(args.opp_noise_ratio)):
                legal = list(board.legal_moves())
                if not legal:
                    move = -1
                    uci = "0000"
                else:
                    move = int(rng.choice(legal))
                    uci = internal_move_to_uci(move)
            else:
                assert pf is not None
                pf.set_position(_pad_fen(rec.opening_fen or Board().fen()), moves=rec.moves_uci)
                if args.opp_depth:
                    best_uci, _ponder = pf.go_depth(int(args.opp_depth), max_wait_s=600.0)
                elif args.opp_nodes:
                    best_uci, _ponder = pf.go_nodes(int(args.opp_nodes), max_wait_s=60.0)
                else:
                    best_uci, _ponder = pf.go_movetime(
                        int(args.opp_movetime_ms or 500),
                        max_wait_s=float(args.opp_movetime_ms or 500) / 1000.0 + 60.0,
                    )
                uci = best_uci[:4]
                move = uci_move_to_internal(uci)

        rec.moves_uci.append(uci)
        board.push(int(move))
        ply += 1

    rec.result = game_result or "draw"
    rec.termination = TERMINATION_LABELS.get(game_termination, str(game_termination))
    rec.plies = ply
    return rec, game_termination, ply


def play_match(args) -> ArenaResult:
    device = torch.device(args.device)

    openings: list[dict[str, Any]] = []
    opening_path = str(getattr(args, "opening_suite_path", "") or "")
    if opening_path:
        openings = _load_opening_suite(
            opening_path,
            max_openings=int(getattr(args, "max_openings", 0)),
        )
    total_games = (
        int(len(openings) * int(args.games_per_opening))
        if openings
        else int(args.games)
    )

    parallel_games = max(1, min(int(args.parallel_games), int(total_games)))
    use_batcher = parallel_games > 1 and bool(args.cross_game_batching)

    if use_batcher:
        # Load model once, wrap in cross-game batcher; all worker threads share it.
        model, cand_step = _build_arena_model(Path(args.checkpoint), args.value_checkpoint)
        model.to(device).eval()
        evaluator = CrossGameBatcher(
            model=model,
            device=device,
            use_bfloat16=not args.disable_bf16,
            max_batch_size=int(args.cross_game_batch_cap),
            coalesce_timeout_ms=float(args.cross_game_coalesce_ms),
        )
    else:
        model, cand_step = _build_arena_model(Path(args.checkpoint), args.value_checkpoint)
        model.to(device)
        model.eval()
        evaluator = make_gpu_evaluator(model, device=str(device), use_bfloat16=not args.disable_bf16)
    base_evaluator = evaluator
    evaluator = _wrap_value_source(base_evaluator, args.our_value_source)
    args._shadow_evaluator = None
    if str(args.our_shadow_value_source) != "none":
        args._shadow_evaluator = _wrap_value_source(base_evaluator, args.our_shadow_value_source)
    args._danger_runtime = None
    if str(args.our_danger_head):
        args._danger_runtime = _DangerHeadRuntime(
            args.our_danger_head,
            device=device,
            use_bfloat16=not args.disable_bf16,
        )
    print(f"loaded our model from step {cand_step}", flush=True)
    print(f"value_source={args.our_value_source}", flush=True)
    if str(args.our_shadow_value_source) != "none":
        shadow_sims = int(args.our_shadow_sims) if int(args.our_shadow_sims) > 0 else int(args.our_sims)
        print(
            f"shadow_value_source={args.our_shadow_value_source} "
            f"sims={shadow_sims} top_k={int(args.our_shadow_top_k)} side={args.our_shadow_side}",
            flush=True,
        )
        if bool(args.our_shadow_disagreement_verifier):
            print(
                f"shadow_disagreement_verifier=on top_k={int(args.our_shadow_verifier_top_k)} "
                f"margin_cp={int(args.our_shadow_verifier_margin_cp)} "
                f"ordinary_min_original_cp={int(args.our_shadow_verifier_ordinary_min_original_cp)} "
                f"mate_risk_margin_cp={int(args.our_shadow_verifier_mate_risk_margin_cp)} "
                f"mate_risk_cp={int(args.our_shadow_verifier_mate_risk_cp)} "
                f"mate_risk_safe_cp={int(args.our_shadow_verifier_mate_risk_safe_cp)} "
                f"escape_margin_cp={int(args.our_shadow_verifier_escape_margin_cp)} "
                f"escape_risk_cp={int(args.our_shadow_verifier_escape_risk_cp)} "
                f"escape_safe_cp={int(args.our_shadow_verifier_escape_safe_cp)} "
                f"trigger_rank={int(args.our_shadow_verifier_trigger_rank)} "
                f"trigger_min_gap={float(args.our_shadow_verifier_trigger_min_gap):.4f} "
                f"ambiguous_gap={float(args.our_shadow_verifier_ambiguous_gap):.4f}",
                flush=True,
            )
    if str(args.our_search) == "alphabeta":
        print(
            f"search=alphabeta depth={int(args.ab_depth)} "
            f"root_branch={int(args.ab_root_max_branch)} "
            f"branch={int(args.ab_max_branch)} "
            f"q_depth={int(args.ab_quiescence_depth)} "
            f"q_branch={int(args.ab_quiescence_max_branch)} "
            f"tt_mb={int(args.ab_tt_mb)}",
            flush=True,
        )
    else:
        print(
            f"search=mcts sims={int(args.our_sims)} c_puct={float(args.our_c_puct):.3f} "
            f"q_weight={float(args.our_q_weight):.3f} temp={float(args.our_temperature_move):.3f}",
            flush=True,
        )
    if args._danger_runtime is not None:
        print(
            f"danger_head={args.our_danger_head} top_k={args.our_danger_top_k} "
            f"mode={args.our_danger_mode} lambda={args.our_danger_lambda:.3f} "
            f"veto={args.our_danger_veto_threshold:.3f} "
            f"triage={args.our_danger_triage_threshold:.3f}/plies{args.our_danger_triage_exact_plies}",
            flush=True,
        )
    if bool(args.our_pikafish_verifier):
        verifier_mode = (
            f"depth={int(args.our_verifier_depth)}" if int(args.our_verifier_depth) > 0
            else f"nodes={int(args.our_verifier_nodes)}" if int(args.our_verifier_nodes) > 0
            else f"movetime_ms={int(args.our_verifier_movetime_ms)}" if int(args.our_verifier_movetime_ms) > 0
            else "depth=1"
        )
        print(
            f"pikafish_verifier=on top_k={int(args.our_verifier_top_k)} "
            f"margin_cp={int(args.our_verifier_margin_cp)} "
            f"danger_threshold_cp={int(args.our_verifier_danger_threshold_cp)} "
            f"side={args.our_verifier_side} "
            f"{verifier_mode}",
            flush=True,
        )
    if use_batcher:
        print(
            f"parallelism: {parallel_games} arena threads + cross-game batcher "
            f"(cap={args.cross_game_batch_cap}, coalesce={args.cross_game_coalesce_ms:.1f}ms)",
            flush=True,
        )
    elif parallel_games > 1:
        print(
            f"parallelism: {parallel_games} arena threads (per-thread evaluators, "
            f"--no-cross-game-batching)",
            flush=True,
        )

    if not args.opp_random:
        strength = ""
        if int(args.opp_uci_elo) > 0:
            strength = f" UCI_LimitStrength=true UCI_Elo={int(args.opp_uci_elo)}"
        elif bool(args.opp_uci_limit_strength):
            strength = " UCI_LimitStrength=true"
        print(
            f"launched pikafish: depth={args.opp_depth} movetime_ms={args.opp_movetime_ms} "
            f"nodes={args.opp_nodes}{strength}  ({parallel_games} per-thread instance(s))",
            flush=True,
        )
    else:
        print("opponent: random-move player (Elo floor baseline)", flush=True)

    if opening_path:
        print(
            f"opening suite: {opening_path} "
            f"openings={len(openings)} games_per_opening={int(args.games_per_opening)}",
            flush=True,
        )

    result = ArenaResult(
        checkpoint=str(Path(args.checkpoint).resolve()),
        games=int(total_games),
        opp_depth=int(args.opp_depth) if args.opp_depth else None,
        opp_movetime_ms=int(args.opp_movetime_ms) if args.opp_movetime_ms else None,
        opp_nodes=int(args.opp_nodes) if args.opp_nodes else None,
        opp_uci_elo=int(args.opp_uci_elo) if int(args.opp_uci_elo) > 0 else None,
        opp_uci_limit_strength=bool(args.opp_uci_limit_strength or int(args.opp_uci_elo) > 0),
        our_sims=int(args.our_sims),
        our_side_counts={"red": 0, "black": 0},
        search_kind=str(args.our_search),
    )

    output_lock = threading.Lock()
    results_by_gi: dict[int, tuple[GameRecord, int, int]] = {}
    worker_errors: list[str] = []
    completed = 0

    def worker_loop(thread_id: int, indices: list[int]) -> None:
        """One thread plays its assigned game indices.  Owns its own engine + RNG."""
        nonlocal completed
        own_pf: PikafishOpponent | None = None
        own_verifier_pf: PikafishOpponent | None = None
        if not args.opp_random:
            if args.opp_engine == "fairy_sf":
                own_pf = make_fairy_stockfish_opponent(
                    binary_path=args.fairy_sf_binary,
                    threads=int(args.opp_threads),
                    hash_mb=int(args.opp_hash_mb),
                )
            elif args.opp_engine == "elephantart":
                own_pf = make_elephantart_opponent(
                    binary_path=args.elephantart_binary,
                    weights_path=args.elephantart_weights,
                    threads=int(args.opp_threads),
                    playouts=int(args.elephantart_playouts),
                )
            elif args.opp_engine == "eleeye":
                _eleeye_opts = [
                    f"setoption name hashsize value {int(args.eleeye_hashsize_mb)}",
                ]
                if args.eleeye_disable_book:
                    _eleeye_opts.append("setoption name usebook value false")
                own_pf = PikafishOpponent(
                    binary_path=args.eleeye_binary,
                    threads=1,
                    hash_mb=int(args.eleeye_hashsize_mb),
                    handshake="ucci",
                    send_threads_and_hash=False,  # ElephantEye doesn't recognize Threads/Hash names
                    extra_setoption_lines=tuple(_eleeye_opts),
                )
            else:
                pikafish_options: list[str] = []
                if bool(args.opp_uci_limit_strength) or int(args.opp_uci_elo) > 0:
                    pikafish_options.append("setoption name UCI_LimitStrength value true")
                if int(args.opp_uci_elo) > 0:
                    pikafish_options.append(f"setoption name UCI_Elo value {int(args.opp_uci_elo)}")
                own_pf = PikafishOpponent(
                    binary_path=args.pikafish_binary,
                    threads=int(args.opp_threads),
                    hash_mb=int(args.opp_hash_mb),
                    extra_setoption_lines=tuple(pikafish_options),
                    required_option_names=(
                        ("UCI_LimitStrength", "UCI_Elo")
                        if int(args.opp_uci_elo) > 0
                        else (("UCI_LimitStrength",) if bool(args.opp_uci_limit_strength) else ())
                    ),
                )
        if bool(args.our_pikafish_verifier) or bool(args.our_shadow_disagreement_verifier):
            own_verifier_pf = PikafishOpponent(
                binary_path=args.pikafish_binary,
                threads=int(args.our_verifier_threads),
                hash_mb=int(args.our_verifier_hash_mb),
            )
        own_rng = random.Random(int(args.seed) + thread_id * 999_983 + 7)
        try:
            try:
                for gi in indices:
                    opening_entry = None
                    opening_index = None
                    if openings:
                        opening_index = int(gi // int(args.games_per_opening))
                        opening_entry = openings[opening_index]
                    rec, term, ply = _play_one_arena_game(
                        gi=gi,
                        args=args,
                        evaluator=evaluator,
                        pf=own_pf,
                        verifier_pf=own_verifier_pf,
                        rng=own_rng,
                        opening_entry=opening_entry,
                        opening_index=opening_index,
                    )
                    with output_lock:
                        results_by_gi[gi] = (rec, term, ply)
                        completed += 1
                        print(
                            f"game {completed}/{total_games} (gi={gi}) "
                            f"our_side={rec.our_side} result={rec.result} "
                            f"plies={ply} term={rec.termination}",
                            flush=True,
                        )
            except BaseException:
                err = traceback.format_exc()
                with output_lock:
                    worker_errors.append(f"thread_id={thread_id}\n{err}")
                print(f"ERROR: arena worker {thread_id} failed", flush=True)
        finally:
            if own_pf is not None:
                try:
                    own_pf.close()
                except Exception:
                    pass
            if own_verifier_pf is not None:
                try:
                    own_verifier_pf.close()
                except Exception:
                    pass

    # Partition games round-robin across threads (so colors stay balanced per thread).
    threads: list[threading.Thread] = []
    for tid in range(parallel_games):
        indices = list(range(tid, int(total_games), parallel_games))
        if not indices:
            continue
        t = threading.Thread(
            target=worker_loop, args=(tid, indices),
            name=f"arena-{tid}", daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    if worker_errors:
        if hasattr(evaluator, "close"):
            evaluator.close()
        raise RuntimeError("arena worker failure(s):\n" + "\n".join(worker_errors))
    if len(results_by_gi) != int(total_games):
        if hasattr(evaluator, "close"):
            evaluator.close()
        raise RuntimeError(
            f"arena produced {len(results_by_gi)}/{int(total_games)} completed games; refusing partial JSON"
        )

    if use_batcher:
        stats = evaluator.stats()
        evaluator.close()
        print(
            f"  cross-game batcher: {stats['batches_run']} GPU batches "
            f"served {stats['calls_received']} thread calls "
            f"(coalesce ratio={stats['coalesce_ratio']:.1f}x, "
            f"avg_leaves/batch={stats['avg_leaves_per_batch']:.1f}, "
            f"max_batch={stats['max_observed_batch']})",
            flush=True,
        )

    # Aggregate in deterministic gi order.
    termination_counts: dict[int, int] = {}
    total_plies = 0
    for gi in sorted(results_by_gi.keys()):
        rec, term, ply = results_by_gi[gi]
        result.per_game.append(rec)
        result.our_side_counts[rec.our_side] += 1
        if rec.result == "our_win":
            result.our_wins += 1
            if rec.our_side == "red":
                result.our_wins_as_red += 1
            else:
                result.our_wins_as_black += 1
        elif rec.result == "opp_win":
            result.opp_wins += 1
        else:
            result.draws += 1
        termination_counts[term] = termination_counts.get(term, 0) + 1
        total_plies += ply

    result.termination_counts = {
        TERMINATION_LABELS.get(k, str(k)): v for k, v in termination_counts.items()
    }
    games_done = result.our_wins + result.opp_wins + result.draws
    result.avg_plies = total_plies / max(games_done, 1)
    result.score_rate = (result.our_wins + 0.5 * result.draws) / max(games_done, 1)
    result.elo_estimate = _elo_from_score(result.score_rate)
    guard_events = [event for rec in result.per_game for event in rec.guard_events]
    events_by_side: dict[str, int] = {}
    events_by_type: dict[str, int] = {}
    events_by_replacement: dict[str, int] = {}
    for event in guard_events:
        side = str(event.get("side", "unknown"))
        events_by_side[side] = events_by_side.get(side, 0) + 1
        guard_type = str(event.get("guard_type", "unknown"))
        events_by_type[guard_type] = events_by_type.get(guard_type, 0) + 1
        replacement = str(event.get("replacement_move_uci", "unknown"))
        events_by_replacement[replacement] = events_by_replacement.get(replacement, 0) + 1
    search_events = [event for rec in result.per_game for event in rec.search_stats]
    search_stat_rows = [
        event.get("stats", {})
        for event in search_events
        if isinstance(event.get("stats", {}), dict) and event.get("stats", {})
    ]
    if search_stat_rows:
        numeric_keys = ["nodes", "leaf_evals", "cutoffs", "tt_hits", "q_nodes", "elapsed_ms"]
        totals: dict[str, float] = {}
        for key in numeric_keys:
            totals[key] = float(sum(float(row.get(key, 0.0) or 0.0) for row in search_stat_rows))
        searches = len(search_stat_rows)
        result.search_stats_summary = {
            "search_kind": str(args.our_search),
            "searches": int(searches),
            "totals": totals,
            "avg_nodes": totals["nodes"] / max(searches, 1),
            "avg_leaf_evals": totals["leaf_evals"] / max(searches, 1),
            "avg_elapsed_ms": totals["elapsed_ms"] / max(searches, 1),
            "max_elapsed_ms": max(float(row.get("elapsed_ms", 0.0) or 0.0) for row in search_stat_rows),
            "max_depth": max(int(row.get("max_depth", 0) or 0) for row in search_stat_rows),
        }
    else:
        result.search_stats_summary = {"search_kind": str(args.our_search), "searches": 0}
    shadow_rows = [
        event.get("shadow_value_probe", {})
        for event in search_events
        if isinstance(event.get("shadow_value_probe", {}), dict)
        and event.get("shadow_value_probe", {})
    ]
    if shadow_rows:
        disagreements = [row for row in shadow_rows if bool(row.get("disagrees_with_actual"))]
        result.shadow_value_summary = {
            "enabled": True,
            "value_source": str(args.our_shadow_value_source),
            "side": str(args.our_shadow_side),
            "sims": int(args.our_shadow_sims) if int(args.our_shadow_sims) > 0 else int(args.our_sims),
            "top_k": int(args.our_shadow_top_k),
            "probes": int(len(shadow_rows)),
            "disagreements": int(len(disagreements)),
            "disagreement_rate": float(len(disagreements) / max(len(shadow_rows), 1)),
            "top_shadow_moves": sorted(
                Counter(str(row.get("shadow_best_move_uci", "")) for row in disagreements).items(),
                key=lambda item: (-item[1], item[0]),
            )[:10],
        }
    else:
        result.shadow_value_summary = {
            "enabled": bool(str(args.our_shadow_value_source) != "none"),
            "value_source": str(args.our_shadow_value_source),
            "probes": 0,
            "disagreements": 0,
        }
    result.symbolic_guard_summary = {
        "search_kind": str(args.our_search),
        "ab_depth": int(args.ab_depth),
        "ab_root_max_branch": int(args.ab_root_max_branch),
        "ab_max_branch": int(args.ab_max_branch),
        "ab_quiescence_depth": int(args.ab_quiescence_depth),
        "ab_quiescence_max_branch": int(args.ab_quiescence_max_branch),
        "ab_tt_mb": int(args.ab_tt_mb),
        "root_mate1_blunder_guard_enabled": bool(args.our_root_mate1_blunder_guard),
        "root_mate2_blunder_guard_enabled": bool(args.our_root_mate2_blunder_guard),
        "root_forcing_check_guard_plies": int(args.our_root_forcing_check_guard_plies),
        "root_forcing_check_guard_max_candidates": int(args.our_root_forcing_check_guard_max_candidates),
        "root_selection_mode": str(args.our_root_selection_mode),
        "root_gumbel_top_k": int(args.our_root_gumbel_top_k),
        "root_gumbel_scale": float(args.our_root_gumbel_scale),
        "log_root_stats_top_k": int(args.our_log_root_stats_top_k),
        "c_puct": float(args.our_c_puct),
        "c_puct_base": float(args.our_c_puct_base),
        "c_puct_factor": float(args.our_c_puct_factor),
        "q_weight": float(args.our_q_weight),
        "q_clip": float(args.our_q_clip),
        "value_source": str(args.our_value_source),
        "shadow_value_source": str(args.our_shadow_value_source),
        "shadow_value_side": str(args.our_shadow_side),
        "shadow_value_sims": int(args.our_shadow_sims) if int(args.our_shadow_sims) > 0 else int(args.our_sims),
        "shadow_value_top_k": int(args.our_shadow_top_k),
        "shadow_disagreement_verifier_enabled": bool(args.our_shadow_disagreement_verifier),
        "shadow_disagreement_verifier_top_k": int(args.our_shadow_verifier_top_k),
        "shadow_disagreement_verifier_margin_cp": int(args.our_shadow_verifier_margin_cp),
        "shadow_disagreement_verifier_ordinary_min_original_cp": int(args.our_shadow_verifier_ordinary_min_original_cp),
        "shadow_disagreement_verifier_mate_risk_margin_cp": int(args.our_shadow_verifier_mate_risk_margin_cp),
        "shadow_disagreement_verifier_mate_risk_cp": int(args.our_shadow_verifier_mate_risk_cp),
        "shadow_disagreement_verifier_mate_risk_safe_cp": int(args.our_shadow_verifier_mate_risk_safe_cp),
        "shadow_disagreement_verifier_escape_margin_cp": int(args.our_shadow_verifier_escape_margin_cp),
        "shadow_disagreement_verifier_escape_risk_cp": int(args.our_shadow_verifier_escape_risk_cp),
        "shadow_disagreement_verifier_escape_safe_cp": int(args.our_shadow_verifier_escape_safe_cp),
        "shadow_disagreement_verifier_trigger_rank": int(args.our_shadow_verifier_trigger_rank),
        "shadow_disagreement_verifier_trigger_min_gap": float(args.our_shadow_verifier_trigger_min_gap),
        "shadow_disagreement_verifier_ambiguous_gap": float(args.our_shadow_verifier_ambiguous_gap),
        "fpu_reduction_root": float(args.our_fpu_reduction_root),
        "fpu_reduction_tree": float(args.our_fpu_reduction_tree),
        "pikafish_verifier_enabled": bool(args.our_pikafish_verifier),
        "pikafish_verifier_top_k": int(args.our_verifier_top_k),
        "pikafish_verifier_margin_cp": int(args.our_verifier_margin_cp),
        "pikafish_verifier_danger_threshold_cp": int(args.our_verifier_danger_threshold_cp),
        "pikafish_verifier_depth": int(args.our_verifier_depth),
        "pikafish_verifier_nodes": int(args.our_verifier_nodes),
        "pikafish_verifier_movetime_ms": int(args.our_verifier_movetime_ms),
        "pikafish_verifier_max_wait_s": float(args.our_verifier_max_wait_s),
        "pikafish_verifier_side": str(args.our_verifier_side),
        "cnn_danger_head_enabled": bool(str(args.our_danger_head)),
        "cnn_danger_mode": str(args.our_danger_mode),
        "cnn_danger_top_k": int(args.our_danger_top_k),
        "cnn_danger_lambda": float(args.our_danger_lambda),
        "cnn_danger_veto_threshold": float(args.our_danger_veto_threshold),
        "cnn_danger_triage_threshold": float(args.our_danger_triage_threshold),
        "cnn_danger_triage_exact_plies": int(args.our_danger_triage_exact_plies),
        "cnn_danger_triage_exact_max_candidates": int(args.our_danger_triage_exact_max_candidates),
        "tactical_mate1_extension_enabled": bool(args.our_tactical_mate1_extension),
        "tactical_mate2_extension_enabled": bool(args.our_tactical_mate2_extension),
        "events": len(guard_events),
        "games_with_events": sum(1 for rec in result.per_game if rec.guard_events),
        "events_by_side": events_by_side,
        "events_by_type": events_by_type,
        "top_replacement_moves": sorted(
            events_by_replacement.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10],
    }

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--value-checkpoint", default="",
                   help="optional second checkpoint supplying value_scalar/wdl_logits "
                        "(policy/value CHIMERA candidate)")
    p.add_argument("--output-dir", default="/home/laure/alphaxiang/external_arena_runs")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--our-side", choices=["alternate", "red", "black"], default="alternate",
                   help="Which side our model plays. Default alternates by game index.")
    p.add_argument("--opening-suite-path", default="",
                   help="Optional JSON opening suite. Supports the xiangqi_arena opening-suite "
                        "format with a top-level positions list.")
    p.add_argument("--games-per-opening", type=int, default=2,
                   help="When --opening-suite-path is set, play this many games from each opening. "
                        "Default 2.")
    p.add_argument("--max-openings", type=int, default=0,
                   help="Limit openings loaded from --opening-suite-path. 0 means all openings.")
    p.add_argument("--parallel-games", type=int, default=8,
                   help="Number of arena games to play concurrently in worker threads. "
                        "Each thread has its own Pikafish + RNG. Set to 1 for serial.")
    p.add_argument("--cross-game-batching", action=argparse.BooleanOptionalAction, default=True,
                   help="Aggregate MCTS leaf evaluations across all parallel games "
                        "into single GPU forward passes (default ON).")
    p.add_argument("--cross-game-batch-cap", type=int, default=256,
                   help="Maximum batch size the cross-game batcher will assemble. Default 256.")
    p.add_argument("--cross-game-coalesce-ms", type=float, default=2.0,
                   help="How long the batcher waits to gather more leaves before forcing "
                        "a partial batch. Default 2.0ms.")
    p.add_argument("--seed", type=int, default=20260418)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable-bf16", action="store_true")

    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--fairy-sf-binary", default="/home/laure/engines/fairy-stockfish-xq",
                   help="Path to Fairy-Stockfish xiangqi binary (used when --opp-engine=fairy_sf)")
    p.add_argument("--elephantart-binary",
                   default="/home/laure/engines/ElephantArt/build/Elephant",
                   help="Path to ElephantArt binary (used when --opp-engine=elephantart)")
    p.add_argument("--elephantart-weights",
                   default="/home/laure/engines/ElephantArt/weights/NN/trained-5-23-45000games-4b-256c.txt",
                   help="ElephantArt weights file (.txt format)")
    p.add_argument("--elephantart-playouts", type=int, default=800,
                   help="ElephantArt MCTS playouts per move (strength knob; ignored for other engines)")
    p.add_argument("--eleeye-binary",
                   default="/home/laure/external_engines/eleeye/eleeye/ELEEYE.EXE",
                   help="Path to ElephantEye binary (used when --opp-engine=eleeye)")
    p.add_argument("--eleeye-hashsize-mb", type=int, default=64,
                   help="ElephantEye hashsize in MB (default 64; engine accepts 16..1024)")
    p.add_argument("--eleeye-disable-book", action=argparse.BooleanOptionalAction, default=True,
                   help="Disable ElephantEye opening book for cleaner playing-strength evaluation (default ON).")
    p.add_argument("--opp-engine", default="pikafish",
                   choices=["pikafish", "fairy_sf", "elephantart", "eleeye"],
                   help="Which engine to use as opponent: "
                        "'pikafish' (NNUE+alphabeta, default), "
                        "'fairy_sf' (Fairy-Stockfish in xiangqi+UCCI mode, different NNUE+alphabeta), "
                        "'elephantart' (AlphaZero-style CNN+MCTS, fundamentally different paradigm), "
                        "'eleeye' (ElephantEye 3.31, classic UCCI alpha-beta engine — public ladder anchor).")
    p.add_argument("--opp-depth", type=int, default=0, help="if >0, use fixed-depth search")
    p.add_argument("--opp-movetime-ms", type=int, default=0, help="if >0, use fixed-time search")
    p.add_argument("--opp-nodes", type=int, default=0, help="if >0, use fixed-node search (weakest knob)")
    p.add_argument("--opp-uci-limit-strength", action=argparse.BooleanOptionalAction, default=False,
                   help="For Pikafish only: enable UCI_LimitStrength. Automatically enabled "
                        "when --opp-uci-elo is set.")
    p.add_argument("--opp-uci-elo", type=int, default=0,
                   help="For Pikafish only: set public UCI_Elo strength. Pikafish documents "
                        "the range as 1280..3133; 0 disables this option.")
    p.add_argument("--opp-random", action="store_true", help="use random-move opponent (no Pikafish)")
    p.add_argument("--opp-noise-ratio", type=float, default=0.0,
                   help="With this probability per move, replace Pikafish's choice with a random "
                        "legal move. 0.0 = pure Pikafish (default). 0.15 matches the Stage-1 "
                        "training-time opponent noise.")
    p.add_argument("--opp-threads", type=int, default=1)
    p.add_argument("--opp-hash-mb", type=int, default=64)

    p.add_argument("--our-search", choices=["mcts", "alphabeta"], default="mcts",
                   help="Search backend for our model. Default mcts preserves legacy behavior.")
    p.add_argument("--our-sims", type=int, default=400)
    p.add_argument("--our-c-puct", type=float, default=1.25)
    p.add_argument("--our-c-puct-base", type=float, default=1.0,
                   help="Base term for optional log-scaled cPUCT. With "
                        "--our-c-puct-factor=0 this has no effect.")
    p.add_argument("--our-c-puct-factor", type=float, default=0.0,
                   help="If non-zero, use c_eff = c_puct + factor * "
                        "log((parent_visits + base + 1) / base). Default 0 "
                        "keeps legacy fixed c_puct.")
    p.add_argument("--our-q-weight", type=float, default=1.0,
                   help="Scale applied to child Q in PUCT. This is the practical "
                        "search-time value-scale knob for v12.5 diagnostics. Default 1.0.")
    p.add_argument("--our-q-clip", type=float, default=1.0,
                   help="Absolute clamp for child Q before --our-q-weight is applied. Default 1.0.")
    p.add_argument("--our-fpu-reduction-root", type=float, default=-1.0,
                   help="Root first-play urgency reduction. Negative keeps legacy "
                        "unvisited-child Q=0; non-negative uses parent_value - reduction.")
    p.add_argument("--our-fpu-reduction-tree", type=float, default=-1.0,
                   help="Tree first-play urgency reduction. Negative keeps legacy "
                        "unvisited-child Q=0; non-negative uses parent_value - reduction.")
    p.add_argument("--our-value-source", choices=["scalar", "wdl"], default="scalar",
                   help="Value source consumed by MCTS. 'scalar' uses value_scalar (default); "
                        "'wdl' uses softmax(wdl_logits)[win] - softmax(wdl_logits)[loss].")
    p.add_argument("--our-shadow-value-source", choices=["none", "scalar", "wdl"], default="none",
                   help="Shadow-only value-source probe. Runs an extra MCTS with this value "
                        "source, records its root choice in JSON, and never changes the move.")
    p.add_argument("--our-shadow-sims", type=int, default=0,
                   help="Simulation count for --our-shadow-value-source. 0 reuses --our-sims.")
    p.add_argument("--our-shadow-top-k", type=int, default=8,
                   help="Number of shadow root visit candidates saved in per-move search_stats.")
    p.add_argument("--our-shadow-side", choices=["any", "red", "black"], default="any",
                   help="Limit shadow value-source probes to our red/black moves. Default any.")
    p.add_argument("--our-shadow-disagreement-verifier", action="store_true", default=False,
                   help="If shadow and actual root moves disagree, verify the scalar/shadow "
                        "top-K union with Pikafish and override only if the verified margin "
                        "exceeds --our-shadow-verifier-margin-cp. Requires a shadow value source.")
    p.add_argument("--our-shadow-verifier-top-k", type=int, default=6,
                   help="Scalar and shadow top-K candidate count for --our-shadow-disagreement-verifier.")
    p.add_argument("--our-shadow-verifier-margin-cp", type=int, default=300,
                   help="Minimum verified child-eval improvement required by "
                        "--our-shadow-disagreement-verifier.")
    p.add_argument("--our-shadow-verifier-ordinary-min-original-cp", type=int, default=-20000,
                   help="Minimum original opponent-pov child eval required for ordinary "
                        "margin-only shadow verifier overrides. Low values preserve legacy "
                        "behavior; positive values prevent tiny low-risk eval gains from "
                        "changing strategic plans.")
    p.add_argument("--our-shadow-verifier-mate-risk-margin-cp", type=int, default=-1,
                   help="Optional lower improvement threshold for shadow disagreement "
                        "overrides when the original move allows a verified mate/high-risk "
                        "child and the replacement does not. Negative disables this split "
                        "threshold and preserves the ordinary margin-only behavior.")
    p.add_argument("--our-shadow-verifier-mate-risk-cp", type=int, default=19000,
                   help="Opponent-pov child eval at or above this centipawn value is treated "
                        "as mate-risk for --our-shadow-verifier-mate-risk-margin-cp.")
    p.add_argument("--our-shadow-verifier-mate-risk-safe-cp", type=int, default=19000,
                   help="Maximum replacement opponent-pov child eval allowed when the original "
                        "move is mate-risk. Values below --our-shadow-verifier-mate-risk-cp "
                        "also constrain ordinary-margin replacements from mate-risk originals. "
                        "Default 19000 preserves legacy behavior.")
    p.add_argument("--our-shadow-verifier-escape-margin-cp", type=int, default=-1,
                   help="Optional gray-zone escape threshold for shadow disagreement overrides. "
                        "When enabled, allow replacement if original child eval is at least "
                        "--our-shadow-verifier-escape-risk-cp, replacement child eval is at most "
                        "--our-shadow-verifier-escape-safe-cp, and improvement reaches this value. "
                        "Negative disables this rule.")
    p.add_argument("--our-shadow-verifier-escape-risk-cp", type=int, default=500,
                   help="Original opponent-pov child eval threshold for the gray-zone escape rule.")
    p.add_argument("--our-shadow-verifier-escape-safe-cp", type=int, default=100,
                   help="Replacement opponent-pov child eval threshold for the gray-zone escape rule.")
    p.add_argument("--our-shadow-verifier-trigger-rank", type=int, default=0,
                   help="Optional extra trigger for --our-shadow-disagreement-verifier: when >0, "
                        "also verify if the actual move is absent from shadow top-K or has a "
                        "shadow rank worse than this value. Default 0 preserves disagreement-only behavior.")
    p.add_argument("--our-shadow-verifier-trigger-min-gap", type=float, default=0.0,
                   help="Minimum shadow visit-prob gap between shadow top-1 and the actual move "
                        "for --our-shadow-verifier-trigger-rank. Default 0.0.")
    p.add_argument("--our-shadow-verifier-ambiguous-gap", type=float, default=-1.0,
                   help="Optional extra trigger for --our-shadow-disagreement-verifier: when >=0, "
                        "also verify if shadow top-1 and top-2 visit probabilities differ by at "
                        "most this value. Default -1 disables this trigger.")
    p.add_argument("--ab-depth", type=int, default=3,
                   help="Alpha-beta plies when --our-search=alphabeta.")
    p.add_argument("--ab-root-max-branch", type=int, default=32,
                   help="Root candidate cap for policy-guided alpha-beta. <=0 means all legal moves.")
    p.add_argument("--ab-max-branch", type=int, default=16,
                   help="Interior candidate cap for policy-guided alpha-beta. <=0 means all legal moves.")
    p.add_argument("--ab-quiescence-depth", type=int, default=1,
                   help="Quiescence extension depth for immediate wins, checks, and captures.")
    p.add_argument("--ab-quiescence-max-branch", type=int, default=8,
                   help="Quiescence candidate cap. <=0 means all tactical candidates.")
    p.add_argument("--ab-tt-mb", type=int, default=128,
                   help="Approximate Python transposition table budget in MB for alpha-beta.")
    p.add_argument("--our-temperature-move", type=float, default=0.1,
                   help="near-arg-max; slightly >0 to break ties")
    p.add_argument("--our-add-root-noise", action="store_true", default=False)
    p.add_argument("--our-dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--our-dirichlet-eps", type=float, default=0.1)
    p.add_argument("--our-root-selection-mode", choices=["visit", "gumbel_visit"], default="visit",
                   help="'visit' uses the C++ MCTS selected move. 'gumbel_visit' is a "
                        "lightweight root-level probe that samples among the top visit "
                        "candidates with log(prob)+Gumbel noise before downstream guards.")
    p.add_argument("--our-root-gumbel-top-k", type=int, default=8,
                   help="Top root visit candidates considered by --our-root-selection-mode=gumbel_visit.")
    p.add_argument("--our-root-gumbel-scale", type=float, default=0.75,
                   help="Gumbel perturbation scale for --our-root-selection-mode=gumbel_visit.")
    p.add_argument("--our-log-root-stats-top-k", type=int, default=0,
                   help="If >0, store MCTS root stats for the top-K visit candidates in "
                        "per-game search_stats. This is intended for d20 data-scale "
                        "collection so later audits do not need to rerun MCTS.")
    p.add_argument("--our-pikafish-verifier", action="store_true", default=False,
                   help="V14V prototype: after our MCTS root search, evaluate top-K "
                        "candidate child positions with Pikafish and override only if "
                        "another candidate is safer by --our-verifier-margin-cp.")
    p.add_argument("--our-verifier-top-k", type=int, default=3,
                   help="Number of root candidates to evaluate with --our-pikafish-verifier.")
    p.add_argument("--our-verifier-margin-cp", type=int, default=120,
                   help="Minimum opponent-POV child-eval improvement required before "
                        "the verifier overrides V13's root choice.")
    p.add_argument("--our-verifier-danger-threshold-cp", type=int, default=-20000,
                   help="Only allow verifier overrides when the original selected "
                        "child eval is at least this bad from the opponent POV. "
                        "Default -20000 preserves old behavior; use e.g. 600 for "
                        "high-risk-only verifier gating.")
    p.add_argument("--our-verifier-depth", type=int, default=5,
                   help="Pikafish verifier depth. If 0, use nodes or movetime.")
    p.add_argument("--our-verifier-nodes", type=int, default=0,
                   help="Pikafish verifier node budget; used only when depth is 0.")
    p.add_argument("--our-verifier-movetime-ms", type=int, default=0,
                   help="Pikafish verifier movetime; used only when depth and nodes are 0.")
    p.add_argument("--our-verifier-max-wait-s", type=float, default=600.0,
                   help="Maximum seconds to wait for each Pikafish verifier child search. "
                        "Depth-20 formal runs should raise this instead of lowering depth.")
    p.add_argument("--our-verifier-threads", type=int, default=1)
    p.add_argument("--our-verifier-hash-mb", type=int, default=64)
    p.add_argument("--our-verifier-side", choices=["any", "red", "black"], default="any",
                   help="Limit verifier to our red/black moves. Default any.")
    p.add_argument("--our-tactical-mate1-extension", action="store_true", default=False,
                   help="Symbolic probe: at neural leaf eval time, if side-to-move has an "
                        "immediate terminal win, back up +1 instead of using value head.")
    p.add_argument("--our-tactical-mate2-extension", action="store_true", default=False,
                   help="Symbolic probe: at neural leaf eval time, if side-to-move has a "
                        "check-only forced mate-in-2, back up +1 instead of using value head. "
                        "Quiet mate nets are intentionally ignored.")
    p.add_argument("--our-root-mate1-blunder-guard", action="store_true", default=False,
                   help="Symbolic guard: if the selected root move gives the opponent an "
                        "immediate terminal win, choose the highest-probability root "
                        "candidate that does not.")
    p.add_argument("--our-root-mate2-blunder-guard", action="store_true", default=False,
                   help="Symbolic guard: if the selected root move gives the opponent a "
                        "check-only forced mate-in-2, choose the highest-probability root "
                        "candidate that does not.")
    p.add_argument("--our-root-forcing-check-guard-plies", type=int, default=0,
                   help="Broader root guard: if >0, refuse selected moves that allow the "
                        "opponent a check-only forced win within this many half-plies. "
                        "Use 5 for mate-in-3 style probes. Default 0 disables it.")
    p.add_argument("--our-root-forcing-check-guard-max-candidates", type=int, default=12,
                   help="Maximum root policy alternatives to test when the selected move "
                        "fails --our-root-forcing-check-guard-plies. Default 12.")
    p.add_argument("--our-danger-head", default="",
                   help="Optional V14D CNN action-danger checkpoint. If set, root top-k "
                        "candidate moves are reranked by predicted tactical danger after "
                        "the candidate move is applied.")
    p.add_argument("--our-danger-mode", choices=["rerank", "triage"], default="rerank",
                   help="'rerank' directly adjusts root top-k by danger score. "
                        "'triage' only uses CNN danger to decide whether to run exact "
                        "mate/forcing guards on the selected move.")
    p.add_argument("--our-danger-top-k", type=int, default=8)
    p.add_argument("--our-danger-lambda", type=float, default=2.0)
    p.add_argument("--our-danger-veto-threshold", type=float, default=0.85)
    p.add_argument("--our-danger-triage-threshold", type=float, default=0.90)
    p.add_argument("--our-danger-triage-exact-plies", type=int, default=5)
    p.add_argument("--our-danger-triage-exact-max-candidates", type=int, default=12)

    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--repeat-limit", type=int, default=6)
    p.add_argument("--repeat-min-ply", type=int, default=30)
    p.add_argument("--no-capture-limit", type=int, default=60)

    args = p.parse_args()
    knobs = [bool(args.opp_depth), bool(args.opp_movetime_ms), bool(args.opp_nodes), bool(args.opp_random)]
    if sum(knobs) != 1:
        p.error("exactly one of --opp-depth / --opp-movetime-ms / --opp-nodes / --opp-random must be set")
    if int(args.opp_uci_elo) and args.opp_engine != "pikafish":
        p.error("--opp-uci-elo is supported only with --opp-engine=pikafish")
    if int(args.opp_uci_elo) and not (1280 <= int(args.opp_uci_elo) <= 3133):
        p.error("--opp-uci-elo must be in Pikafish's documented 1280..3133 range")
    if int(args.games_per_opening) < 1:
        p.error("--games-per-opening must be >= 1")
    if int(args.max_openings) < 0:
        p.error("--max-openings must be >= 0")
    if str(args.our_search) == "alphabeta":
        if int(args.ab_depth) < 1:
            p.error("--ab-depth must be >= 1")
        if int(args.ab_root_max_branch) < 0:
            p.error("--ab-root-max-branch must be >= 0")
        if int(args.ab_max_branch) < 0:
            p.error("--ab-max-branch must be >= 0")
        if int(args.ab_quiescence_depth) < 0:
            p.error("--ab-quiescence-depth must be >= 0")
        if int(args.ab_quiescence_max_branch) < 0:
            p.error("--ab-quiescence-max-branch must be >= 0")
        if int(args.ab_tt_mb) < 0:
            p.error("--ab-tt-mb must be >= 0")
    verifier_knobs = [int(args.our_verifier_depth) > 0, int(args.our_verifier_nodes) > 0, int(args.our_verifier_movetime_ms) > 0]
    if bool(args.our_pikafish_verifier) and sum(verifier_knobs) > 1:
        p.error("for --our-pikafish-verifier, use only one of verifier depth/nodes/movetime")
    return args


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    result = play_match(args)
    dt = time.monotonic() - t0

    payload = asdict(result)
    payload["duration_s"] = dt
    payload["config"] = {
        k: getattr(args, k)
        for k in vars(args)
        if not str(k).startswith("_")
    }
    payload["config"]["checkpoint"] = str(Path(args.checkpoint).resolve())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"external_arena_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("", flush=True)
    print("=" * 60, flush=True)
    print(
        f"DONE: {result.our_wins}W - {result.opp_wins}L - {result.draws}D "
        f"over {result.games} games",
        flush=True,
    )
    print(f"  score_rate = {result.score_rate:.3f}", flush=True)
    print(f"  elo_estimate vs opponent = {result.elo_estimate:+.0f}", flush=True)
    print(f"  avg_plies = {result.avg_plies:.1f}", flush=True)
    print(f"  termination_counts = {result.termination_counts}", flush=True)
    print(f"  search_stats_summary = {result.search_stats_summary}", flush=True)
    print(f"  symbolic_guard_summary = {result.symbolic_guard_summary}", flush=True)
    print(f"  duration = {dt:.1f}s", flush=True)
    print(f"  saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
