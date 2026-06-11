"""Headless tournament: current best Transformer vs the legacy CNN baseline.

Both sides drive the SAME C++ MCTS extension (`xiangqi_mcts_ext.mcts_search`); they
just plug in different evaluators (`make_gpu_evaluator` for the transformer,
`LegacyCnnEvaluator` for the CNN).

Default settings match `tools/external_arena.py` so the W/L/D number is comparable
to our other arena measurements.

Usage:
    python tools/transformer_vs_cnn_arena.py \
        --transformer-checkpoint /home/laure/alphaxiang/training_runs/run_002_pikafish_curriculum/best.pt \
        --cnn-engine CNN/Chessv11_cpp_hist8_115_mps_fp16.py \
        --cnn-weights CNN/best.pth \
        --games 70
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Block pygame from trying to talk to a display server when we import the gui module
# (it brings pygame in transitively).  SDL "dummy" driver is enough for headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_mcts_ext import Board, mcts_search  # noqa: E402
from xiangqi_model_battle_gui import (  # noqa: E402  reuse the existing loaders
    AgentSpec,
    load_cnn_agent,
    load_transformer_agent,
)


TERMINAL_ONGOING = -1


def play_one_game(
    *,
    transformer_agent: AgentSpec,
    cnn_agent: AgentSpec,
    transformer_is_red: bool,
    sims: int,
    c_puct: float,
    temperature_move: float,
    add_root_noise: bool,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    eval_batch_size: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
    max_plies: int,
    seed_base: int,
) -> dict:
    """Play one full game between the two agents.  Alternates moves until terminal."""
    board = Board()
    moves: list[int] = []
    ply = 0

    while True:
        term = int(board.terminal_code(
            max_plies, repeat_limit, repeat_min_ply, no_capture_limit,
        ))
        if term != TERMINAL_ONGOING:
            break

        red_to_move = (int(board.turn()) == 0)
        agent = transformer_agent if (red_to_move == transformer_is_red) else cnn_agent

        best_move, _, _, _ = mcts_search(
            board=board,
            net=agent.evaluator,
            num_simulations=int(sims),
            c_puct=float(c_puct),
            q_weight=1.0,
            q_clip=1.0,
            add_root_noise=bool(add_root_noise),
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_eps=float(dirichlet_eps),
            temperature_move=float(temperature_move),
            temperature_target=1.0,
            eval_batch_size=int(eval_batch_size),
            seed=int(seed_base + ply * 31),
            canonical_input=True,
            canonical_policy=True,
            max_plies=max_plies,
            repeat_limit=repeat_limit,
            repeat_min_ply=repeat_min_ply,
            no_capture_limit=no_capture_limit,
        )
        if int(best_move) < 0:
            break
        moves.append(int(best_move))
        board.push(int(best_move))
        ply += 1

    term_code = int(board.terminal_code(
        max_plies, repeat_limit, repeat_min_ply, no_capture_limit,
    ))
    red_result = int(board.terminal_result_red_view(term_code))
    if red_result == 0:
        outcome = "draw"
    elif (red_result > 0) == transformer_is_red:
        outcome = "transformer_win"
    else:
        outcome = "cnn_win"

    return {
        "transformer_is_red": transformer_is_red,
        "plies": ply,
        "outcome": outcome,
        "termination_code": term_code,
        "final_red_result": red_result,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transformer-checkpoint", required=True)
    p.add_argument("--cnn-engine", required=True,
                   help="path to CNN engine .py file (e.g. CNN/Chessv11_cpp_hist8_115_mps_fp16.py)")
    p.add_argument("--cnn-weights", required=True,
                   help="path to CNN .pth weights file (e.g. CNN/best.pth)")
    p.add_argument("--output-dir", default="/home/laure/alphaxiang/arena_runs/transformer_vs_cnn")
    p.add_argument("--games", type=int, default=70)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sims", type=int, default=400, help="MCTS sims per move (both sides)")
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--temperature-move", type=float, default=0.1)
    p.add_argument("--add-root-noise", action="store_true",
                   help="Inject Dirichlet root noise (off by default — matches external_arena)")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.1)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--repeat-limit", type=int, default=6)
    p.add_argument("--repeat-min-ply", type=int, default=30)
    p.add_argument("--no-capture-limit", type=int, default=60)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260424)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading transformer: {args.transformer_checkpoint}", flush=True)
    transformer_agent = load_transformer_agent(Path(args.transformer_checkpoint), args.device)
    print(f"  -> {transformer_agent.label}", flush=True)

    print(f"loading CNN: engine={args.cnn_engine}  weights={args.cnn_weights}", flush=True)
    cnn_agent = load_cnn_agent(
        Path(args.cnn_engine), Path(args.cnn_weights), args.device,
    )
    print(f"  -> {cnn_agent.label}", flush=True)

    print(f"running {args.games} games  sims={args.sims}  device={args.device}", flush=True)
    print(f"transformer plays red on even-indexed games (0,2,4,...)", flush=True)
    print()

    t_start = time.monotonic()
    games: list[dict] = []
    transformer_wins = cnn_wins = draws = 0

    for gi in range(int(args.games)):
        transformer_is_red = (gi % 2 == 0)
        seed_base = int(args.seed) + gi * 9_007
        t0 = time.monotonic()
        result = play_one_game(
            transformer_agent=transformer_agent,
            cnn_agent=cnn_agent,
            transformer_is_red=transformer_is_red,
            sims=args.sims,
            c_puct=args.c_puct,
            temperature_move=args.temperature_move,
            add_root_noise=args.add_root_noise,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            eval_batch_size=args.eval_batch_size,
            repeat_limit=args.repeat_limit,
            repeat_min_ply=args.repeat_min_ply,
            no_capture_limit=args.no_capture_limit,
            max_plies=args.max_plies,
            seed_base=seed_base,
        )
        dt = time.monotonic() - t0
        if result["outcome"] == "transformer_win":
            transformer_wins += 1
        elif result["outcome"] == "cnn_win":
            cnn_wins += 1
        else:
            draws += 1
        games.append({"index": gi, **result, "duration_s": dt})

        elapsed = time.monotonic() - t_start
        print(
            f"game {gi+1}/{args.games}  "
            f"trans={'red' if transformer_is_red else 'black'}  "
            f"plies={result['plies']:3d}  "
            f"outcome={result['outcome']:<16}  "
            f"running T-C-D={transformer_wins}-{cnn_wins}-{draws}  "
            f"dt={dt:.0f}s  total={elapsed:.0f}s",
            flush=True,
        )

    total = transformer_wins + cnn_wins + draws
    score_rate = (transformer_wins + 0.5 * draws) / max(1, total)
    decisive = transformer_wins + cnn_wins
    decisive_winrate = transformer_wins / max(1, decisive)
    dt_total = time.monotonic() - t_start

    summary = {
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "cnn_weights": str(args.cnn_weights),
        "games": int(args.games),
        "sims": int(args.sims),
        "device": args.device,
        "transformer_wins": transformer_wins,
        "cnn_wins": cnn_wins,
        "draws": draws,
        "score_rate": score_rate,
        "decisive_winrate": decisive_winrate,
        "duration_s": dt_total,
        "config": vars(args),
        "per_game": games,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = output_dir / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 64)
    print(f"FINAL: Transformer {transformer_wins}W - {cnn_wins}L - {draws}D / {total}")
    print(f"  score_rate (draws=0.5):  {score_rate*100:.1f}%")
    print(f"  decisive winrate (T/(T+C)): {decisive_winrate*100:.1f}%")
    print(f"  duration: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"  saved to: {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
