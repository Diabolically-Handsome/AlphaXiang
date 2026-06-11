#!/usr/bin/env python3
"""Policy-guided selective alpha-beta search for AlphaXiang diagnostics.

This module intentionally keeps policy logits out of the final score. The
network policy is used only to order and cap moves; leaf values come from the
model value head.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from xiangqi_mcts_ext import Board, canonical_action


Evaluator = Callable[[torch.Tensor], dict[str, torch.Tensor]]
TERMINAL_ONGOING = -1


@dataclass
class AlphaBetaConfig:
    depth: int = 3
    root_max_branch: int = 32
    max_branch: int = 16
    quiescence_depth: int = 1
    quiescence_max_branch: int = 8
    tt_mb: int = 128
    max_plies: int = 300
    repeat_limit: int = 6
    repeat_min_ply: int = 30
    no_capture_limit: int = 60


@dataclass
class AlphaBetaStats:
    nodes: int = 0
    leaf_evals: int = 0
    cutoffs: int = 0
    tt_hits: int = 0
    q_nodes: int = 0
    elapsed_ms: float = 0.0
    root_candidates: int = 0
    max_depth: int = 0


class _Search:
    def __init__(self, board: Board, evaluator: Evaluator, config: AlphaBetaConfig):
        self.board = board
        self.evaluator = evaluator
        self.config = config
        self.stats = AlphaBetaStats()
        # Exact-value TT only. The entry count is intentionally conservative;
        # Python object overhead is large and this diagnostic path should stay
        # bounded even during deeper selective searches.
        self.tt_max_entries = max(0, int(config.tt_mb) * 1024 * 1024 // 256)
        self.tt: dict[tuple[int, int, int, int, int, int, int], float] = {}

    def terminal_code(self) -> int:
        return int(
            self.board.terminal_code(
                int(self.config.max_plies),
                int(self.config.repeat_limit),
                int(self.config.repeat_min_ply),
                int(self.config.no_capture_limit),
            )
        )

    def terminal_value_for_side_to_move(self, term: int) -> float:
        if term == TERMINAL_ONGOING:
            raise ValueError("terminal_value_for_side_to_move called on ongoing board")
        red_result = int(self.board.terminal_result_red_view(term))
        if red_result == 0:
            return 0.0
        side_is_red = int(self.board.turn()) == 0
        side_won = (red_result > 0 and side_is_red) or (red_result < 0 and not side_is_red)
        return 1.0 if side_won else -1.0

    def tt_key(self, depth: int, q_depth: int) -> tuple[int, int, int, int, int, int, int]:
        return (
            int(self.board.key()),
            int(self.board.turn()),
            int(self.board.no_capture_count()),
            int(self.board.current_repetition_count()),
            int(self.board.plies_played()),
            int(depth),
            int(q_depth),
        )

    def store_tt(self, key: tuple[int, int, int, int, int, int, int], value: float) -> None:
        if self.tt_max_entries <= 0:
            return
        if len(self.tt) >= self.tt_max_entries:
            self.tt.clear()
        self.tt[key] = float(value)

    def evaluate(self) -> tuple[float, torch.Tensor]:
        x = self.board.to_tensor_canonical()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32, device="cpu").contiguous()
        with torch.no_grad():
            out = self.evaluator(x)
        value = float(out["value_scalar"].detach().flatten()[0].clamp(-1.0, 1.0).item())
        logits = out["policy_logits"].detach().flatten().to(device="cpu", dtype=torch.float32)
        self.stats.leaf_evals += 1
        return value, logits

    def move_gives_check(self, move: int) -> bool:
        self.board.push(int(move))
        try:
            return bool(self.board.in_check_turn())
        finally:
            self.board.pop()

    def move_is_immediate_win(self, move: int) -> bool:
        self.board.push(int(move))
        try:
            term = self.terminal_code()
            if term == TERMINAL_ONGOING:
                return False
            # After a push, the side to move is the opponent. A negative child
            # terminal value means the mover just won.
            return self.terminal_value_for_side_to_move(term) < -0.5
        finally:
            self.board.pop()

    def order_moves(
        self,
        legal_moves: list[int],
        policy_logits: torch.Tensor,
        cap: int,
        tactical_only: bool = False,
    ) -> list[int]:
        stm_black = bool(int(self.board.turn()) == 1)
        scored: list[tuple[float, int]] = []
        for mv in legal_moves:
            mv_i = int(mv)
            is_capture = bool(self.board.is_capture(mv_i))
            gives_check = self.move_gives_check(mv_i)
            immediate_win = self.move_is_immediate_win(mv_i)
            if tactical_only and not (immediate_win or gives_check or is_capture):
                continue
            idx = int(canonical_action(mv_i, stm_black))
            policy_score = float(policy_logits[idx].item()) if 0 <= idx < policy_logits.numel() else 0.0
            bonus = 0.0
            if immediate_win:
                bonus += 1_000_000.0
            if gives_check:
                bonus += 0.25
            if is_capture:
                bonus += 0.15
            scored.append((policy_score + bonus, mv_i))
        scored.sort(key=lambda item: item[0], reverse=True)
        if cap > 0:
            scored = scored[: int(cap)]
        return [mv for _, mv in scored]

    def quiesce(self, alpha: float, beta: float, q_depth: int) -> float:
        self.stats.q_nodes += 1
        term = self.terminal_code()
        if term != TERMINAL_ONGOING:
            return self.terminal_value_for_side_to_move(term)

        stand_pat, logits = self.evaluate()
        if q_depth <= 0:
            return stand_pat
        if stand_pat >= beta:
            self.stats.cutoffs += 1
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        legal = [int(m) for m in self.board.legal_moves()]
        moves = self.order_moves(
            legal,
            logits,
            int(self.config.quiescence_max_branch),
            tactical_only=True,
        )
        best = stand_pat
        for mv in moves:
            self.board.push(mv)
            try:
                score = -self.quiesce(-beta, -alpha, q_depth - 1)
            finally:
                self.board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.stats.cutoffs += 1
                break
        return best

    def negamax(self, depth: int, alpha: float, beta: float, q_depth: int) -> float:
        self.stats.nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, int(depth))

        term = self.terminal_code()
        if term != TERMINAL_ONGOING:
            return self.terminal_value_for_side_to_move(term)

        key = self.tt_key(depth, q_depth)
        cached = self.tt.get(key)
        if cached is not None:
            self.stats.tt_hits += 1
            return cached

        if depth <= 0:
            return self.quiesce(alpha, beta, q_depth)

        node_value, logits = self.evaluate()
        legal = [int(m) for m in self.board.legal_moves()]
        if not legal:
            self.store_tt(key, node_value)
            return node_value

        moves = self.order_moves(legal, logits, int(self.config.max_branch), tactical_only=False)
        best = -math.inf
        cutoff = False
        for mv in moves:
            self.board.push(mv)
            try:
                score = -self.negamax(depth - 1, -beta, -alpha, q_depth)
            finally:
                self.board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.stats.cutoffs += 1
                cutoff = True
                break

        if not cutoff:
            self.store_tt(key, best)
        return float(best)

    def root(self) -> tuple[int, torch.Tensor, torch.Tensor, float, dict[str, object]]:
        t0 = time.perf_counter()
        term = self.terminal_code()
        if term != TERMINAL_ONGOING:
            stats = asdict(self.stats)
            stats["elapsed_ms"] = 0.0
            stats["terminal_root"] = True
            stats["config"] = asdict(self.config)
            return -1, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32), 0.0, stats

        _, logits = self.evaluate()
        legal = [int(m) for m in self.board.legal_moves()]
        moves = self.order_moves(legal, logits, int(self.config.root_max_branch), tactical_only=False)
        self.stats.root_candidates = len(moves)
        if not moves:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stats = asdict(self.stats)
            stats["elapsed_ms"] = elapsed_ms
            stats["config"] = asdict(self.config)
            return -1, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32), 0.0, stats

        root_stm_black = bool(int(self.board.turn()) == 1)
        scored: list[tuple[int, float]] = []
        alpha = -math.inf
        beta = math.inf
        child_depth = max(0, int(self.config.depth) - 1)
        for mv in moves:
            self.board.push(mv)
            try:
                score = -self.negamax(
                    child_depth,
                    -beta,
                    -alpha,
                    max(0, int(self.config.quiescence_depth)),
                )
            finally:
                self.board.pop()
            scored.append((mv, float(score)))
            if score > alpha:
                alpha = score

        scored.sort(key=lambda item: item[1], reverse=True)
        best_move, root_value = scored[0]
        idxs = torch.tensor(
            [int(canonical_action(mv, root_stm_black)) for mv, _ in scored],
            dtype=torch.long,
        )
        values = torch.tensor([score for _, score in scored], dtype=torch.float32)
        if values.numel() == 0:
            probs = torch.empty(0, dtype=torch.float32)
        elif values.numel() == 1:
            probs = torch.ones(1, dtype=torch.float32)
        else:
            probs = torch.softmax(values * 4.0, dim=0).to(dtype=torch.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.elapsed_ms = elapsed_ms
        stats = asdict(self.stats)
        stats["config"] = asdict(self.config)
        stats["root_value"] = float(root_value)
        stats["root_best_move"] = int(best_move)
        stats["root_values"] = [{"move": int(mv), "value": float(score)} for mv, score in scored[:16]]
        return int(best_move), idxs, probs, float(root_value), stats


def alpha_beta_search(
    board: Board,
    evaluator: Evaluator,
    config: AlphaBetaConfig,
) -> tuple[int, torch.Tensor, torch.Tensor, float, dict[str, object]]:
    """Run policy-guided selective alpha-beta from the current board."""

    search = _Search(board, evaluator, config)
    return search.root()
