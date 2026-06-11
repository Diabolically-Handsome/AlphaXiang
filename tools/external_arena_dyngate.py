"""External arena DYNGATE variant: same as external_arena.py but with dynamic
phase-based q_weight gating in MCTS.

v12.6-micro Path C: q_weight is computed per move based on game phase (ply count),
allowing different value-vs-policy balance in opening / mid-game / endgame.

Phase-based q_weight default:
  ply <= --our-q-weight-mid-ply (default 30):  --our-q-weight-open (default 1.0)
  ply <= --our-q-weight-end-ply (default 80):  --our-q-weight-mid  (default 1.2)
  ply  > --our-q-weight-end-ply:                --our-q-weight-end  (default 1.5)

Defaults chosen to be slightly conservative (small per-phase deltas from baseline 1.0).
Setting all three to the same value reproduces fixed-q_weight behavior.

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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TOOLS = _REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import random  # noqa: E402

from cross_game_batcher import CrossGameBatcher  # noqa: E402
from elephantart_opponent import make_elephantart_opponent  # noqa: E402
from fairy_stockfish_opponent import make_fairy_stockfish_opponent  # noqa: E402
from pikafish_opponent import (  # noqa: E402
    PikafishOpponent,
    internal_move_to_uci,
    uci_move_to_internal,
)
from xiangqi_mcts_ext import Board, make_gpu_evaluator, mcts_search  # noqa: E402
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


@dataclass
class GameRecord:
    index: int
    our_side: str  # "red" or "black"
    moves_uci: list[str] = field(default_factory=list)
    result: str = ""  # "our_win", "opp_win", "draw"
    termination: str = ""
    plies: int = 0
    opening_fen: str = ""


@dataclass
class ArenaResult:
    checkpoint: str
    games: int
    opp_depth: int | None
    opp_movetime_ms: int | None
    opp_nodes: int | None
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


def _load_model(checkpoint_path: Path, device: torch.device, use_bfloat16: bool = True):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    evaluator = make_gpu_evaluator(model, device=str(device), use_bfloat16=use_bfloat16)
    step = int(state.get("global_step", 0))
    return evaluator, step


def _compute_q_weight_for_ply(
    ply: int,
    *,
    open_qw: float, mid_qw: float, end_qw: float,
    mid_ply: int, end_ply: int,
) -> float:
    """v12.6-micro dyngate: phase-based q_weight."""
    if ply <= int(mid_ply):
        return float(open_qw)
    if ply <= int(end_ply):
        return float(mid_qw)
    return float(end_qw)


def _our_pick_move(
    board: Board,
    evaluator,
    *,
    num_simulations: int,
    c_puct: float,
    temperature_move: float,
    q_weight: float,
    q_clip: float,
    add_root_noise: bool,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    seed: int,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> int:
    best_move, _idxs, _probs, _root_v = mcts_search(
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
    )
    return int(best_move)


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
    rng: random.Random,
) -> tuple[GameRecord, int, int]:
    """Play one full game against the configured opponent.  Returns (record, term_code, plies).

    Pure per-game function — no shared state mutation.  Caller is responsible for
    threading-safe accumulation of GameRecords + counts.
    """
    our_is_red = (gi % 2 == 0)
    our_side = "red" if our_is_red else "black"
    board = Board()
    rec = GameRecord(index=gi, our_side=our_side, opening_fen=board.fen())

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
            # v12.6-micro dyngate: compute q_weight from current ply
            dyn_qw = _compute_q_weight_for_ply(
                ply,
                open_qw=args.our_q_weight_open,
                mid_qw=args.our_q_weight_mid,
                end_qw=args.our_q_weight_end,
                mid_ply=args.our_q_weight_mid_ply,
                end_ply=args.our_q_weight_end_ply,
            )
            move = _our_pick_move(
                board,
                evaluator,
                num_simulations=args.our_sims,
                c_puct=args.our_c_puct,
                temperature_move=args.our_temperature_move,
                q_weight=dyn_qw,
                q_clip=args.our_q_clip,
                add_root_noise=bool(args.our_add_root_noise),
                dirichlet_alpha=args.our_dirichlet_alpha,
                dirichlet_eps=args.our_dirichlet_eps,
                seed=int(args.seed + gi * 10_007 + ply * 31),
                max_plies=args.max_plies,
                repeat_limit=args.repeat_limit,
                repeat_min_ply=args.repeat_min_ply,
                no_capture_limit=args.no_capture_limit,
            )
            if move < 0:
                game_termination = TERMINAL_CHECKMATE_OR_STALEMATE
                red_result = int(board.terminal_result_red_view(game_termination))
                game_result = "draw" if red_result == 0 else (
                    "our_win" if (red_result > 0) == our_is_red else "opp_win"
                )
                break
            uci = internal_move_to_uci(move)
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
                pf.set_position(_pad_fen(Board().fen()), moves=rec.moves_uci)
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

    parallel_games = max(1, min(int(args.parallel_games), int(args.games)))
    use_batcher = parallel_games > 1 and bool(args.cross_game_batching)

    if use_batcher:
        # Load model once, wrap in cross-game batcher; all worker threads share it.
        state = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        model = build_model_from_checkpoint_state(state)
        model.to(device).eval()
        cand_step = int(state.get("global_step", 0))
        evaluator = CrossGameBatcher(
            model=model,
            device=device,
            use_bfloat16=not args.disable_bf16,
            max_batch_size=int(args.cross_game_batch_cap),
            coalesce_timeout_ms=float(args.cross_game_coalesce_ms),
        )
    else:
        evaluator, cand_step = _load_model(
            Path(args.checkpoint), device, use_bfloat16=not args.disable_bf16,
        )
    print(f"loaded our model from step {cand_step}", flush=True)
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
        print(
            f"launched pikafish: depth={args.opp_depth} movetime_ms={args.opp_movetime_ms} "
            f"nodes={args.opp_nodes}  ({parallel_games} per-thread instance(s))",
            flush=True,
        )
    else:
        print("opponent: random-move player (Elo floor baseline)", flush=True)

    result = ArenaResult(
        checkpoint=str(Path(args.checkpoint).resolve()),
        games=int(args.games),
        opp_depth=int(args.opp_depth) if args.opp_depth else None,
        opp_movetime_ms=int(args.opp_movetime_ms) if args.opp_movetime_ms else None,
        opp_nodes=int(args.opp_nodes) if args.opp_nodes else None,
        our_sims=int(args.our_sims),
        our_side_counts={"red": 0, "black": 0},
    )

    output_lock = threading.Lock()
    results_by_gi: dict[int, tuple[GameRecord, int, int]] = {}
    completed = 0

    def worker_loop(thread_id: int, indices: list[int]) -> None:
        """One thread plays its assigned game indices.  Owns its own engine + RNG."""
        nonlocal completed
        own_pf: PikafishOpponent | None = None
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
                own_pf = PikafishOpponent(
                    binary_path=args.pikafish_binary,
                    threads=int(args.opp_threads),
                    hash_mb=int(args.opp_hash_mb),
                )
        own_rng = random.Random(int(args.seed) + thread_id * 999_983 + 7)
        try:
            for gi in indices:
                rec, term, ply = _play_one_arena_game(
                    gi=gi, args=args, evaluator=evaluator, pf=own_pf, rng=own_rng,
                )
                with output_lock:
                    results_by_gi[gi] = (rec, term, ply)
                    completed += 1
                    print(
                        f"game {completed}/{args.games} (gi={gi}) "
                        f"our_side={rec.our_side} result={rec.result} "
                        f"plies={ply} term={rec.termination}",
                        flush=True,
                    )
        finally:
            if own_pf is not None:
                try:
                    own_pf.close()
                except Exception:
                    pass

    # Partition games round-robin across threads (so colors stay balanced per thread).
    threads: list[threading.Thread] = []
    for tid in range(parallel_games):
        indices = list(range(tid, int(args.games), parallel_games))
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

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="/home/laure/alphaxiang/external_arena_runs")
    p.add_argument("--games", type=int, default=10)
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
    p.add_argument("--opp-random", action="store_true", help="use random-move opponent (no Pikafish)")
    p.add_argument("--opp-noise-ratio", type=float, default=0.0,
                   help="With this probability per move, replace Pikafish's choice with a random "
                        "legal move. 0.0 = pure Pikafish (default). 0.15 matches the Stage-1 "
                        "training-time opponent noise.")
    p.add_argument("--opp-threads", type=int, default=1)
    p.add_argument("--opp-hash-mb", type=int, default=64)

    p.add_argument("--our-sims", type=int, default=400)
    p.add_argument("--our-c-puct", type=float, default=1.25)
    p.add_argument("--our-q-weight", type=float, default=1.0,
                   help="Legacy fixed q_weight (v12.6 dyngate ignores this in favor "
                        "of --our-q-weight-{open,mid,end}). Kept for arg compatibility.")
    # v12.6-micro Path C: phase-based dyngate
    p.add_argument("--our-q-weight-open", type=float, default=1.0,
                   help="q_weight for opening (ply <= --our-q-weight-mid-ply). Default 1.0.")
    p.add_argument("--our-q-weight-mid", type=float, default=1.2,
                   help="q_weight for middle game. Default 1.2 (slight value boost).")
    p.add_argument("--our-q-weight-end", type=float, default=1.5,
                   help="q_weight for endgame (ply > --our-q-weight-end-ply). Default 1.5.")
    p.add_argument("--our-q-weight-mid-ply", type=int, default=30,
                   help="Opening / mid-game boundary (ply count). Default 30.")
    p.add_argument("--our-q-weight-end-ply", type=int, default=80,
                   help="Mid-game / endgame boundary (ply count). Default 80.")
    p.add_argument("--our-q-clip", type=float, default=1.0,
                   help="Absolute clamp for child Q before --our-q-weight is applied. Default 1.0.")
    p.add_argument("--our-temperature-move", type=float, default=0.1,
                   help="near-arg-max; slightly >0 to break ties")
    p.add_argument("--our-add-root-noise", action="store_true", default=False)
    p.add_argument("--our-dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--our-dirichlet-eps", type=float, default=0.1)

    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--repeat-limit", type=int, default=6)
    p.add_argument("--repeat-min-ply", type=int, default=30)
    p.add_argument("--no-capture-limit", type=int, default=60)

    args = p.parse_args()
    knobs = [bool(args.opp_depth), bool(args.opp_movetime_ms), bool(args.opp_nodes), bool(args.opp_random)]
    if sum(knobs) != 1:
        p.error("exactly one of --opp-depth / --opp-movetime-ms / --opp-nodes / --opp-random must be set")
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
    payload["config"] = {k: getattr(args, k) for k in vars(args)}
    payload["config"]["checkpoint"] = str(Path(args.checkpoint).resolve())
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"external_arena_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"DONE: {result.our_wins}W - {result.opp_wins}L - {result.draws}D over {args.games} games", flush=True)
    print(f"  score_rate = {result.score_rate:.3f}", flush=True)
    print(f"  elo_estimate vs opponent = {result.elo_estimate:+.0f}", flush=True)
    print(f"  avg_plies = {result.avg_plies:.1f}", flush=True)
    print(f"  termination_counts = {result.termination_counts}", flush=True)
    print(f"  duration = {dt:.1f}s", flush=True)
    print(f"  saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
