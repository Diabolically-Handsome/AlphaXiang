"""Headless tournament: two Transformer checkpoints play head-to-head.

Both sides drive the same C++ MCTS extension (`xiangqi_mcts_ext.mcts_search`); they
just plug in different evaluators built from the two checkpoint paths.

Modeled directly on `transformer_vs_cnn_arena.py` — this is the model-vs-model
analogue used for "cyber dueling" comparisons (v7 vs v9, v10 vs v7, etc.).

Sides alternate red/black per game: agent A plays red on even-indexed games (0,2,...)
and black on odd-indexed games. Final reports A-B-D from agent A's perspective.

Usage:
    python tools/transformer_vs_transformer_arena.py \\
        --a-checkpoint /home/laure/alphaxiang/PEAK_step255000_v10_probe3_score77pct_d1.pt \\
        --b-checkpoint /home/laure/alphaxiang/PEAK_step232500_v7_probe23_score72pct_d1.pt \\
        --games 50 --sims 800 --device cuda:0 \\
        --output-dir /home/laure/alphaxiang/arena_runs/v10_vs_v7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_mcts_ext import Board, mcts_search  # noqa: E402
from xiangqi_model_battle_gui import (  # noqa: E402
    AgentSpec,
    load_transformer_agent,
)


TERMINAL_ONGOING = -1


def play_one_game(
    *,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    a_is_red: bool,
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
    """Play one full game between the two agents. Alternates moves until terminal."""
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
        agent = agent_a if (red_to_move == a_is_red) else agent_b

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
    elif (red_result > 0) == a_is_red:
        outcome = "a_win"
    else:
        outcome = "b_win"

    return {
        "a_is_red": a_is_red,
        "plies": ply,
        "outcome": outcome,
        "termination_code": term_code,
        "final_red_result": red_result,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--a-checkpoint", required=True, help="agent A transformer checkpoint")
    p.add_argument("--b-checkpoint", required=True, help="agent B transformer checkpoint")
    p.add_argument("--a-label", default=None, help="display label for A (default: derived from path)")
    p.add_argument("--b-label", default=None, help="display label for B (default: derived from path)")
    p.add_argument("--output-dir", default="/home/laure/alphaxiang/arena_runs/transformer_vs_transformer")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sims", type=int, default=800)
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
    p.add_argument("--seed", type=int, default=20260429)
    return p.parse_args()


def _short_label(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).stem
    if "v10" in name:
        return "v10"
    if "v9" in name:
        return "v9"
    if "v8" in name:
        return "v8"
    if "v7" in name:
        return "v7"
    if "v6" in name:
        return "v6"
    if "v5" in name:
        return "v5"
    if "v4" in name:
        return "v4"
    return name[:24]


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_label = args.a_label or _short_label(args.a_checkpoint)
    b_label = args.b_label or _short_label(args.b_checkpoint)

    print(f"loading A ({a_label}): {args.a_checkpoint}", flush=True)
    agent_a = load_transformer_agent(Path(args.a_checkpoint), args.device)
    print(f"  -> {agent_a.label}", flush=True)

    print(f"loading B ({b_label}): {args.b_checkpoint}", flush=True)
    agent_b = load_transformer_agent(Path(args.b_checkpoint), args.device)
    print(f"  -> {agent_b.label}", flush=True)

    print(f"running {args.games} games  sims={args.sims}  device={args.device}", flush=True)
    print(f"{a_label} plays red on even-indexed games (0,2,4,...)", flush=True)
    print()

    t_start = time.monotonic()
    games: list[dict] = []
    a_wins = b_wins = draws = 0

    for gi in range(int(args.games)):
        a_is_red = (gi % 2 == 0)
        seed_base = int(args.seed) + gi * 9_007
        t0 = time.monotonic()
        result = play_one_game(
            agent_a=agent_a,
            agent_b=agent_b,
            a_is_red=a_is_red,
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
        if result["outcome"] == "a_win":
            a_wins += 1
        elif result["outcome"] == "b_win":
            b_wins += 1
        else:
            draws += 1
        games.append({"index": gi, **result, "duration_s": dt})

        elapsed = time.monotonic() - t_start
        print(
            f"game {gi+1}/{args.games}  "
            f"{a_label}={'red' if a_is_red else 'black'}  "
            f"plies={result['plies']:3d}  "
            f"outcome={result['outcome']:<10}  "
            f"running {a_label}-{b_label}-D={a_wins}-{b_wins}-{draws}  "
            f"dt={dt:.0f}s  total={elapsed:.0f}s",
            flush=True,
        )

    total = a_wins + b_wins + draws
    score_rate = (a_wins + 0.5 * draws) / max(1, total)
    decisive = a_wins + b_wins
    decisive_winrate = a_wins / max(1, decisive)
    dt_total = time.monotonic() - t_start

    summary = {
        "a_checkpoint": str(args.a_checkpoint),
        "b_checkpoint": str(args.b_checkpoint),
        "a_label": a_label,
        "b_label": b_label,
        "games": int(args.games),
        "sims": int(args.sims),
        "device": args.device,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "score_rate_a": score_rate,
        "decisive_winrate_a": decisive_winrate,
        "duration_s": dt_total,
        "config": vars(args),
        "per_game": games,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = output_dir / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 64)
    print(f"FINAL: {a_label} {a_wins}W - {b_wins}L - {draws}D / {total}  (vs {b_label})")
    print(f"  score_rate {a_label} (draws=0.5):  {score_rate*100:.1f}%")
    print(f"  decisive winrate ({a_label}/({a_label}+{b_label})): {decisive_winrate*100:.1f}%")
    print(f"  duration: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"  saved to: {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
