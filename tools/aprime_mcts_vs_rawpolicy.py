"""A' diagnostic: does geo's MCTS@N beat geo's RAW POLICY (no search)?

If MCTS does NOT convincingly beat the raw policy, the self-play training target
(the MCTS visit distribution) is no better than the policy it trains -> regression
explained ("teacher-depth floor"), without needing to express d20 in sims.

MCTS side : geo @ N sims, root Dirichlet noise ON (matches production gate; also
            supplies game-to-game variation so we get a win RATE, not one game).
Raw side  : geo @ 1 sim (= argmax of the policy prior, no search benefit),
            root noise OFF (pure policy).
Both sides use the SAME network (geo). Only the search budget differs.
Reuses the exact arena primitives (Board, mcts_search, loaders) -- production
arena code is untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# this script lives outside the repo; make the repo importable (xiangqi_mcts_ext etc.)
sys.path.insert(0, "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer")

from xiangqi_mcts_ext import Board, make_gpu_evaluator, mcts_search
from xiangqi_selfplay import _load_model_from_checkpoint
from xiangqi_arena import TERMINAL_ONGOING, _termination_label

# arena-default board / search constants (kept identical to xiangqi_arena.ArenaConfig)
MAX_PLIES = 240
REPEAT_LIMIT = 6
REPEAT_MIN_PLY = 30
NO_CAPTURE_LIMIT = 60
C_PUCT = 1.25
Q_WEIGHT = 1.0
Q_CLIP = 1.0
EVAL_BATCH = 16
TEMP_MOVE = 1e-6
DIR_ALPHA = 0.30
DIR_EPS = 0.10
TERM_CHECKMATE = 0  # terminal_result fallback code; only used for failed-search edge case


def _terminal_code(board: Board) -> int:
    return int(board.terminal_code(MAX_PLIES, REPEAT_LIMIT, REPEAT_MIN_PLY, NO_CAPTURE_LIMIT))


def play_game(net, red_sims: int, black_sims: int, red_noise: bool, black_noise: bool, seed: int):
    """Return (result_red_view, termination_code, plies). +1 red win, -1 black win, 0 draw."""
    board = Board()
    while True:
        ply = int(board.plies_played())
        tcode = _terminal_code(board)
        if tcode != TERMINAL_ONGOING:
            return int(board.terminal_result_red_view(tcode)), tcode, ply

        stm = int(board.turn())
        sims = red_sims if stm == 0 else black_sims
        noise = red_noise if stm == 0 else black_noise
        try:
            best_move, _idxs, _probs, _root_v = mcts_search(
                board=board,
                net=net,
                num_simulations=int(sims),
                c_puct=C_PUCT,
                q_weight=Q_WEIGHT,
                q_clip=Q_CLIP,
                add_root_noise=bool(noise),
                dirichlet_alpha=DIR_ALPHA,
                dirichlet_eps=DIR_EPS,
                temperature_move=TEMP_MOVE,
                temperature_target=1.0,
                eval_batch_size=EVAL_BATCH,
                seed=int((seed + ply * 10007) & 0x7FFFFFFF),
                canonical_input=True,
                canonical_policy=True,
                max_plies=MAX_PLIES,
                repeat_limit=REPEAT_LIMIT,
                repeat_min_ply=REPEAT_MIN_PLY,
                no_capture_limit=NO_CAPTURE_LIMIT,
            )
        except Exception as exc:
            # side to move failed -> it loses
            winner_red_view = -1 if stm == 0 else 1
            print(f"    [search error stm={stm}: {exc}]", flush=True)
            return winner_red_view, TERM_CHECKMATE, ply

        if int(best_move) < 0:
            tcode2 = _terminal_code(board)
            if tcode2 == TERMINAL_ONGOING:
                # no legal move = side to move loses
                return (-1 if stm == 0 else 1), TERM_CHECKMATE, ply
            return int(board.terminal_result_red_view(tcode2)), tcode2, ply

        board.push(int(best_move))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/laure/alphaxiang/training_runs/run_063_attn_geo/latest.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sims", default="800,3200,8000,12800", help="comma list of N for the MCTS side")
    ap.add_argument("--games", type=int, default=30, help="games per N (colors split evenly)")
    ap.add_argument("--raw-sims", type=int, default=1, help="sims for the raw-policy side")
    ap.add_argument("--seed", type=int, default=70007)
    ap.add_argument("--out", default="/home/laure/alphaxiang/aprime_geo_result.json")
    args = ap.parse_args()

    sims_list = [int(x) for x in args.sims.split(",") if x.strip()]
    print(f"[A'] loading geo: {args.checkpoint}", flush=True)
    model, cfg = _load_model_from_checkpoint(Path(args.checkpoint))
    print(f"[A'] model geometry: use_2d={getattr(cfg,'use_2d_relative_attention_bias',None)} "
          f"use_los={getattr(cfg,'use_line_of_sight_attention_bias',None)}", flush=True)
    net = make_gpu_evaluator(model, device=args.device, use_bfloat16=True)

    all_results = []
    for N in sims_list:
        wins = draws = losses = 0  # from MCTS side's perspective
        plies_sum = 0
        terms: dict[str, int] = {}
        t0 = time.time()
        print(f"\n[A'] === MCTS@{N} sims  vs  raw-policy@{args.raw_sims} ({args.games} games) ===", flush=True)
        for g in range(args.games):
            mcts_is_red = (g % 2 == 0)
            if mcts_is_red:
                rv, tcode, plies = play_game(net, N, args.raw_sims, True, False, args.seed + g * 101)
                mcts_view = rv
            else:
                rv, tcode, plies = play_game(net, args.raw_sims, N, False, True, args.seed + g * 101)
                mcts_view = -rv
            if mcts_view > 0:
                wins += 1; res = "W"
            elif mcts_view < 0:
                losses += 1; res = "L"
            else:
                draws += 1; res = "D"
            plies_sum += plies
            tl = _termination_label(int(tcode))
            terms[tl] = terms.get(tl, 0) + 1
            played = wins + draws + losses
            score = (wins + 0.5 * draws) / played
            print(f"    game {g+1:>3}/{args.games}  mcts={'R' if mcts_is_red else 'B'}  {res}  "
                  f"plies={plies:>3} term={tl:<4}  running score={score*100:5.1f}%  (W{wins}-D{draws}-L{losses})",
                  flush=True)
        played = wins + draws + losses
        score = (wins + 0.5 * draws) / played if played else 0.0
        rec = {
            "sims": N, "raw_sims": args.raw_sims, "games": played,
            "mcts_wins": wins, "draws": draws, "mcts_losses": losses,
            "mcts_score_rate": score, "avg_plies": plies_sum / played if played else 0.0,
            "terminations": terms, "seconds": round(time.time() - t0, 1),
        }
        all_results.append(rec)
        print(f"[A'] >>> MCTS@{N}: score={score*100:.1f}%  W{wins}-D{draws}-L{losses}  "
              f"avg_plies={rec['avg_plies']:.1f}  ({rec['seconds']}s)", flush=True)

    Path(args.out).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("\n[A'] ===== SUMMARY: MCTS@N vs raw policy =====", flush=True)
    print(f"{'N sims':>8} | {'score':>7} | {'W-D-L':>10} | {'avg plies':>9}", flush=True)
    for r in all_results:
        print(f"{r['sims']:>8} | {r['mcts_score_rate']*100:6.1f}% | "
              f"{str(r['mcts_wins'])+'-'+str(r['draws'])+'-'+str(r['mcts_losses']):>10} | {r['avg_plies']:>9.1f}",
              flush=True)
    print(f"[A'] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
