"""Head-to-head diagnostic arena with optional CHIMERA (spliced-head) players.

Forensic experiments (2026-06-09):
  (1) chimera gates @12800: (geo policy + candidate value) vs geo, and
      (candidate policy + geo value) vs geo  ->  WHICH HEAD carries the regression?
  (2) low-sims re-arena: candidate vs geo at {800, 3200}  ->  does less search
      shrink the gap (value-error-amplified-by-deep-search signature)?

Protocol mirrors the production gate (xiangqi_arena.ArenaConfig): both sides same
sims, c_puct=1.25, q_weight=1.0, q_clip=1.0, root Dirichlet noise alpha=0.30
eps=0.10 on BOTH sides, temperature_move=1e-6, max_plies=240, repeat 6/30,
no-capture 60, bf16 eval, colors balanced (candidate is Red on even games).
Unlike the production gate, --seed is required so every arena gets an
independent noise stream (fixes the constant-seed weakness).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer")

from xiangqi_mcts_ext import Board, make_gpu_evaluator, mcts_search
from xiangqi_selfplay import _load_model_from_checkpoint
from xiangqi_arena import TERMINAL_ONGOING, _termination_label

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


class ChimeraModel(nn.Module):
    """Policy head from one model, value (and wdl) head from another."""

    def __init__(self, policy_model: nn.Module, value_model: nn.Module) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, x):
        out_p = self.policy_model(x)
        out_v = self.value_model(x)
        result = dict(out_p)
        result["value_scalar"] = out_v["value_scalar"]
        if "wdl_logits" in out_v:
            result["wdl_logits"] = out_v["wdl_logits"]
        return result


def _terminal_code(board: Board) -> int:
    return int(board.terminal_code(MAX_PLIES, REPEAT_LIMIT, REPEAT_MIN_PLY, NO_CAPTURE_LIMIT))


def play_game(net_red, net_black, sims: int, seed: int):
    """Return (result_red_view, termination_code, plies)."""
    board = Board()
    while True:
        ply = int(board.plies_played())
        tcode = _terminal_code(board)
        if tcode != TERMINAL_ONGOING:
            return int(board.terminal_result_red_view(tcode)), tcode, ply
        stm = int(board.turn())
        net = net_red if stm == 0 else net_black
        try:
            best_move, _i, _p, _v = mcts_search(
                board=board,
                net=net,
                num_simulations=int(sims),
                c_puct=C_PUCT,
                q_weight=Q_WEIGHT,
                q_clip=Q_CLIP,
                add_root_noise=True,
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
            print(f"    [search error stm={stm}: {exc}]", flush=True)
            return (-1 if stm == 0 else 1), 0, ply
        if int(best_move) < 0:
            tcode2 = _terminal_code(board)
            if tcode2 == TERMINAL_ONGOING:
                return (-1 if stm == 0 else 1), 0, ply
            return int(board.terminal_result_red_view(tcode2)), tcode2, ply
        board.push(int(best_move))


def load_net(policy_ckpt: str, value_ckpt: str, device: str):
    pm, pcfg = _load_model_from_checkpoint(Path(policy_ckpt))
    print(f"  policy: {policy_ckpt}", flush=True)
    print(f"    geometry: 2d={getattr(pcfg,'use_2d_relative_attention_bias',None)} "
          f"los={getattr(pcfg,'use_line_of_sight_attention_bias',None)}", flush=True)
    if value_ckpt == policy_ckpt:
        model = pm
        print("  value : (same checkpoint)", flush=True)
    else:
        vm, vcfg = _load_model_from_checkpoint(Path(value_ckpt))
        print(f"  value : {value_ckpt}", flush=True)
        print(f"    geometry: 2d={getattr(vcfg,'use_2d_relative_attention_bias',None)} "
              f"los={getattr(vcfg,'use_line_of_sight_attention_bias',None)}", flush=True)
        model = ChimeraModel(pm, vm)
    return make_gpu_evaluator(model, device=device, use_bfloat16=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--cand-policy", required=True)
    ap.add_argument("--cand-value", default=None, help="default: same as --cand-policy")
    ap.add_argument("--champ", required=True)
    ap.add_argument("--champ-value", default=None, help="default: same as --champ")
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand_value = args.cand_value or args.cand_policy
    champ_value = args.champ_value or args.champ
    print(f"\n[H2H] === {args.label} | sims={args.sims} games={args.games} "
          f"seed={args.seed} dev={args.device} ===", flush=True)
    print("[H2H] candidate:", flush=True)
    cand = load_net(args.cand_policy, cand_value, args.device)
    print("[H2H] champion:", flush=True)
    champ = load_net(args.champ, champ_value, args.device)

    w = d = l = 0
    plies_sum = 0
    terms: dict[str, int] = {}
    t0 = time.time()
    for g in range(args.games):
        cand_is_red = (g % 2 == 0)
        gseed = args.seed + g * 9973
        if cand_is_red:
            rv, tcode, plies = play_game(cand, champ, args.sims, gseed)
            cv = rv
        else:
            rv, tcode, plies = play_game(champ, cand, args.sims, gseed)
            cv = -rv
        if cv > 0:
            w += 1; res = "W"
        elif cv < 0:
            l += 1; res = "L"
        else:
            d += 1; res = "D"
        plies_sum += plies
        tl = _termination_label(int(tcode))
        terms[tl] = terms.get(tl, 0) + 1
        n = w + d + l
        score = (w + 0.5 * d) / n
        print(f"  game {g+1:>3}/{args.games}  cand={'R' if cand_is_red else 'B'}  {res}  "
              f"plies={plies:>3} term={tl:<4}  running={score*100:5.1f}%  (W{w}-D{d}-L{l})",
              flush=True)
    n = w + d + l
    rec = {
        "label": args.label, "sims": args.sims, "games": n, "seed": args.seed,
        "cand_policy": args.cand_policy, "cand_value": cand_value,
        "champ": args.champ, "champ_value": champ_value,
        "wins": w, "draws": d, "losses": l,
        "score_rate": (w + 0.5 * d) / n if n else 0.0,
        "avg_plies": plies_sum / n if n else 0.0,
        "terminations": terms, "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = []
    existing.append(rec)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[H2H] >>> {args.label}: score={rec['score_rate']*100:.1f}%  W{w}-D{d}-L{l}  "
          f"avg_plies={rec['avg_plies']:.1f}  ({rec['seconds']}s)", flush=True)
    print(f"[H2H] appended to {args.out}", flush=True)


if __name__ == "__main__":
    main()
