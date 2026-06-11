"""v12.6-lite Day 1: value/policy three-way diagnostic on arena failure positions.

Extracts our-turn positions from arena JSONs (typically Pika d=4/5/6/7 losses),
then for each position records:
  - v12 model raw outputs: value_scalar, policy top-5 (idx + prob), policy entropy
  - Pikafish d=12 multipv=5 ground truth: best move, top-5 moves, eval cp
  - Whether v12 policy top-1/3/5 contains Pikafish's best move
  - Value calibration: |v12_value - tanh(pikafish_cp / 400)|

Output: JSONL file, one record per position.

Three-way diagnostic questions answered downstream:
  Q1 (policy):  what % of positions have Pika best in v12 top-1/3/5?
  Q2 (value):   what's the value MAE vs tanh-cp-target? saturation rate?
  Q3 (MCTS):    deferred — comparing chosen vs policy top-1 is approx; a true
                Q3 needs running MCTS at sims=1600 per position (expensive).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_opponent import uci_move_to_internal  # noqa: E402
from pikafish_pool import PikafishJob, PikafishPool, PikafishResult  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402
from hard_position_mining import _load_model, predict_values_and_logits  # noqa: E402


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


def _internal_to_uci(internal: int, board: Board) -> str:
    """Convert raw move int to UCI string by trying legal moves."""
    for m in board.legal_moves():
        if int(m) == int(internal):
            return board.uci_move(int(m))
    return f"<unknown:{internal}>"


def _extract_positions(
    arena_json_paths: list[Path],
    *,
    wanted_results: set[str],
    only_our_turns: bool,
    max_plies: int,
    max_positions: int | None,
) -> list[dict[str, Any]]:
    """Extract position records (with state tensor + bookkeeping) from arena JSONs."""
    out: list[dict[str, Any]] = []
    for arena_path in arena_json_paths:
        payload = json.loads(arena_path.read_text(encoding="utf-8"))
        opp_depth = payload.get("opp_depth")
        timestamp = payload.get("timestamp", "")
        for rec in payload.get("per_game", []):
            if str(rec.get("result", "")) not in wanted_results:
                continue
            board = Board()
            opening_fen = str(rec.get("opening_fen") or "")
            if opening_fen:
                board.set_fen(_pad_fen(opening_fen))
            our_is_red = str(rec.get("our_side", "red")) == "red"
            moves = [str(m)[:4] for m in rec.get("moves_uci", [])]
            for ply, uci in enumerate(moves[:max_plies]):
                red_to_move = int(board.turn()) == 0
                our_turn = (red_to_move == our_is_red)
                # Always parse the actual move played (used to advance board)
                raw_move = int(uci_move_to_internal(uci))
                if (not only_our_turns) or our_turn:
                    stm_is_black = bool(board.turn() == 1)
                    state = board.to_tensor_canonical().to(torch.float32)[0].contiguous().clone()
                    legal_raw = list(board.legal_moves())
                    legal_canonical = [int(canonical_action(int(m), stm_is_black))
                                       for m in legal_raw]
                    chosen_canonical = int(canonical_action(raw_move, stm_is_black))
                    if chosen_canonical in legal_canonical:
                        out.append({
                            "state": state,
                            "fen": _pad_fen(board.fen()),
                            "stm_is_black": stm_is_black,
                            "our_side": rec.get("our_side"),
                            "ply": int(ply),
                            "game_index": int(rec.get("index", -1)),
                            "result": str(rec.get("result", "")),
                            "termination": str(rec.get("termination", "")),
                            "source_arena": str(arena_path),
                            "source_timestamp": timestamp,
                            "opp_depth": opp_depth,
                            "chosen_uci": uci,
                            "chosen_canonical": chosen_canonical,
                            "legal_canonical": legal_canonical,
                            "legal_raw": [int(m) for m in legal_raw],
                        })
                        if max_positions is not None and len(out) >= max_positions:
                            return out
                if not bool(board.is_legal(raw_move)):
                    break
                board.push_legal(raw_move)
    return out


def _v12_forward(
    model,
    positions: list[dict[str, Any]],
    device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack states, run forward, return (values (N,), policy_logits (N, 8100))."""
    states = torch.stack([p["state"] for p in positions], dim=0)
    return predict_values_and_logits(model, states, device, batch_size=batch_size)


def _policy_top_k(logits: torch.Tensor, legal_canonical: list[int], k: int = 5) -> tuple[list[int], list[float], float, list[int]]:
    """Mask to legal, softmax, return (top-k canonical idxs, top-k probs, entropy_nats, tied_top1_set).

    tied_top1_set is the set of canonical idxs whose prob equals (within 1e-5) the max prob.
    This is robust to multiple moves having tied softmax probabilities.
    """
    mask = torch.full_like(logits, fill_value=-1e9)
    for idx in legal_canonical:
        if 0 <= idx < mask.numel():
            mask[idx] = 0.0
    masked = logits + mask
    probs = torch.softmax(masked, dim=-1)
    # Entropy over legal subset
    legal_probs = probs[torch.tensor(legal_canonical, dtype=torch.int64)]
    legal_probs = legal_probs.clamp_min(1e-12)
    entropy = float(-(legal_probs * legal_probs.log()).sum().item())
    # Top-k
    top_probs, top_idxs = torch.topk(probs, min(k, probs.numel()))
    top5_idx = [int(i) for i in top_idxs.tolist()]
    top5_prob = [float(p) for p in top_probs.tolist()]
    # Tied top-1 set: anyone within 1e-5 of max prob
    max_p = float(top5_prob[0]) if top5_prob else 0.0
    tied_top1 = [int(i) for i, p in zip(top5_idx, top5_prob) if abs(p - max_p) < 1e-5]
    return top5_idx, top5_prob, entropy, tied_top1


def _query_pikafish(
    positions: list[dict[str, Any]],
    pool: PikafishPool,
    depth: int,
    multipv: int,
    timeout_s: float,
) -> list[dict[str, Any] | None]:
    """Submit one job per position, collect multipv results."""
    jobs = [
        PikafishJob(
            index=i,
            fen=p["fen"],
            depth=int(depth),
            multipv=int(multipv),
        )
        for i, p in enumerate(positions)
    ]
    pool.submit_all(jobs)
    results: list[PikafishResult] = pool.collect(len(jobs), timeout_s=timeout_s)
    by_index: dict[int, PikafishResult] = {r.index: r for r in results}
    out: list[dict[str, Any] | None] = []
    for i in range(len(positions)):
        r = by_index.get(i)
        if r is None or r.error:
            out.append(None)
            continue
        # PikafishResult fields: best_move (UCI), eval_cp (root score),
        # multipv_moves (list[(uci, cp)] for top-k)
        mpv = getattr(r, "multipv_moves", None) or []
        top_k = [{"uci": m[0], "score_cp": int(m[1])} for m in mpv[:multipv]]
        # Fallback: if no multipv_moves, at least record the single best
        if not top_k and getattr(r, "best_move", None):
            top_k = [{"uci": r.best_move, "score_cp": int(getattr(r, "eval_cp", 0))}]
        out.append({
            "eval_cp": int(getattr(r, "eval_cp", 0)),
            "best_move_uci": getattr(r, "best_move", None),
            "top_k": top_k,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arena_json", nargs="+", help="One or more external_arena_*.json files")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--results", default="opp_win",
                   help="Comma-separated arena results: opp_win,our_win,draw")
    p.add_argument("--only-our-turns", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--max-positions", type=int, default=None,
                   help="Cap total positions; useful for smoke tests")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pika-depth", type=int, default=12)
    p.add_argument("--pika-multipv", type=int, default=5)
    p.add_argument("--pika-workers", type=int, default=8)
    p.add_argument("--pika-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--pika-timeout-s", type=float, default=3600.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    t0 = time.monotonic()
    arena_paths = [Path(s) for s in args.arena_json]
    wanted = {x.strip() for x in args.results.split(",") if x.strip()}
    print(f"Extracting positions from {len(arena_paths)} arena JSON(s) "
          f"(results={sorted(wanted)})...", flush=True)
    positions = _extract_positions(
        arena_paths,
        wanted_results=wanted,
        only_our_turns=bool(args.only_our_turns),
        max_plies=int(args.max_plies),
        max_positions=int(args.max_positions) if args.max_positions else None,
    )
    print(f"  extracted {len(positions)} positions", flush=True)
    if not positions:
        raise SystemExit("no positions extracted")

    print(f"Loading model from {args.checkpoint}...", flush=True)
    device = torch.device(args.device)
    model = _load_model(Path(args.checkpoint), device)

    print(f"Running v12 forward pass (batch_size={args.batch_size})...", flush=True)
    values, logits = _v12_forward(model, positions, device, batch_size=int(args.batch_size))

    # Free model memory while Pikafish runs
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Querying Pikafish d={args.pika_depth} multipv={args.pika_multipv} "
          f"with {args.pika_workers} workers...", flush=True)
    pool = PikafishPool(
        num_workers=int(args.pika_workers),
        binary_path=args.pika_binary,
        threads_per_worker=1,
        hash_mb=64,
    )
    try:
        pika_results = _query_pikafish(
            positions, pool,
            depth=int(args.pika_depth),
            multipv=int(args.pika_multipv),
            timeout_s=float(args.pika_timeout_s),
        )
    finally:
        pool.close()

    print("Writing diagnostic JSONL...", flush=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_with_pika = 0
    n_pika_best_in_tied_top1 = 0  # tied-aware top-1 hit
    n_pika_best_in_top3 = 0
    n_pika_best_in_top5 = 0
    n_chosen_in_v12_top1_set = 0  # MCTS chose what policy top-1 (set incl ties) said
    n_chosen_pika_disagree = 0    # MCTS chose != pika best
    value_errors_t400 = []
    value_errors_t200 = []
    value_signs_correct = 0
    value_saturated = 0
    cp_v_pairs = []  # for correlation analysis
    entropy_samples = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, pos in enumerate(positions):
            v = float(values[i].item())
            top5_idx, top5_prob, entropy, tied_top1 = _policy_top_k(
                logits[i], pos["legal_canonical"], k=5
            )
            entropy_samples.append(entropy)
            chosen_in_top1_set = pos["chosen_canonical"] in tied_top1
            if chosen_in_top1_set:
                n_chosen_in_v12_top1_set += 1

            pika = pika_results[i]
            pika_best_in_tied_top1 = pika_best_in_top5 = pika_best_in_top3 = None
            value_target_t400 = value_target_t200 = None
            if pika is not None:
                n_with_pika += 1
                pika_best_uci = pika.get("best_move_uci")
                pika_best_canon: int | None = None
                if pika_best_uci:
                    try:
                        raw = int(uci_move_to_internal(pika_best_uci))
                        pika_best_canon = int(canonical_action(raw, bool(pos["stm_is_black"])))
                        pika_best_in_tied_top1 = pika_best_canon in tied_top1
                        pika_best_in_top3 = pika_best_canon in top5_idx[:3]
                        pika_best_in_top5 = pika_best_canon in top5_idx[:5]
                        if pika_best_in_tied_top1:
                            n_pika_best_in_tied_top1 += 1
                        if pika_best_in_top3:
                            n_pika_best_in_top3 += 1
                        if pika_best_in_top5:
                            n_pika_best_in_top5 += 1
                        # Did our actually-chosen move differ from pika's best?
                        if pos["chosen_canonical"] != pika_best_canon:
                            n_chosen_pika_disagree += 1
                    except Exception:
                        pass
                cp = pika.get("eval_cp")
                if cp is not None:
                    target_400 = math.tanh(float(cp) / 400.0)
                    target_200 = math.tanh(float(cp) / 200.0)
                    value_target_t400 = target_400
                    value_target_t200 = target_200
                    value_errors_t400.append(abs(v - target_400))
                    value_errors_t200.append(abs(v - target_200))
                    cp_v_pairs.append((float(cp), v))
                    # Sign correctness: do v and cp agree on who's winning?
                    if (v > 0 and cp > 0) or (v < 0 and cp < 0) or (abs(v) < 0.05 and abs(cp) < 50):
                        value_signs_correct += 1
                    if abs(v) > 0.95:
                        value_saturated += 1

            rec = {
                "ply": pos["ply"],
                "fen": pos["fen"],
                "stm_is_black": pos["stm_is_black"],
                "our_side": pos["our_side"],
                "game_index": pos["game_index"],
                "result": pos["result"],
                "termination": pos["termination"],
                "opp_depth": pos["opp_depth"],
                "chosen_uci": pos["chosen_uci"],
                "chosen_canonical": pos["chosen_canonical"],
                "chosen_in_v12_top1_set": chosen_in_top1_set,
                "legal_count": len(pos["legal_canonical"]),
                "v12_value": v,
                "v12_policy_entropy_nats": entropy,
                "v12_top5_canonical": top5_idx,
                "v12_top5_prob": top5_prob,
                "v12_tied_top1": tied_top1,
                "pika_eval_cp": pika["eval_cp"] if pika else None,
                "pika_best_uci": pika["best_move_uci"] if pika else None,
                "pika_top_k": pika["top_k"] if pika else None,
                "pika_best_in_v12_tied_top1": pika_best_in_tied_top1,
                "pika_best_in_v12_top3": pika_best_in_top3,
                "pika_best_in_v12_top5": pika_best_in_top5,
                "value_target_tanh400": value_target_t400,
                "value_target_tanh200": value_target_t200,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % int(args.log_every) == 0:
                print(f"  wrote {i+1}/{len(positions)} records", flush=True)

    # Pearson correlation of cp vs v
    pearson_corr = None
    if len(cp_v_pairs) >= 5:
        n = len(cp_v_pairs)
        sx = sum(p[0] for p in cp_v_pairs)
        sy = sum(p[1] for p in cp_v_pairs)
        sxx = sum(p[0]**2 for p in cp_v_pairs)
        syy = sum(p[1]**2 for p in cp_v_pairs)
        sxy = sum(p[0]*p[1] for p in cp_v_pairs)
        num = n*sxy - sx*sy
        den_sq = (n*sxx - sx*sx) * (n*syy - sy*sy)
        if den_sq > 0:
            pearson_corr = num / math.sqrt(den_sq)

    # Entropy distribution
    entropy_samples.sort()
    n_e = len(entropy_samples)
    entropy_p = {q: float(entropy_samples[min(int(n_e*q), n_e-1)]) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if n_e else {}

    summary = {
        "checkpoint": args.checkpoint,
        "n_positions": len(positions),
        "n_with_pika": n_with_pika,
        "Q1_policy_pika_recall": {
            "tied_top1_rate": (n_pika_best_in_tied_top1 / n_with_pika) if n_with_pika else None,
            "top3_rate": (n_pika_best_in_top3 / n_with_pika) if n_with_pika else None,
            "top5_rate": (n_pika_best_in_top5 / n_with_pika) if n_with_pika else None,
            "comment": "tied_top1 = pika best is in v12's set of probability-max moves",
        },
        "Q2_value": {
            "n_with_target": len(value_errors_t400),
            "mae_tanh400": (sum(value_errors_t400) / len(value_errors_t400)) if value_errors_t400 else None,
            "mae_tanh200": (sum(value_errors_t200) / len(value_errors_t200)) if value_errors_t200 else None,
            "sign_correct_rate": (value_signs_correct / len(value_errors_t400)) if value_errors_t400 else None,
            "saturation_rate": (value_saturated / n_with_pika) if n_with_pika else None,
            "pearson_cp_v": pearson_corr,
        },
        "Q3_mcts_choice": {
            "chosen_in_v12_policy_top1_set_rate": n_chosen_in_v12_top1_set / len(positions),
            "chosen_disagrees_with_pika_best_rate": (n_chosen_pika_disagree / n_with_pika) if n_with_pika else None,
            "comment": "chosen = MCTS-selected move from the actual game; if often disagrees with policy top-1 set, MCTS is overriding policy.",
        },
        "policy_entropy_nats": entropy_p,
        "elapsed_s": time.monotonic() - t0,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {len(positions)} positions in {summary['elapsed_s']:.0f}s", flush=True)
    q1 = summary["Q1_policy_pika_recall"]
    print(f"  Q1 pika-best recall in v12: tied_top1={q1['tied_top1_rate']:.3f}  "
          f"top3={q1['top3_rate']:.3f}  top5={q1['top5_rate']:.3f}",
          flush=True)
    q2 = summary["Q2_value"]
    print(f"  Q2 value: MAE_t400={q2['mae_tanh400']:.4f}  MAE_t200={q2['mae_tanh200']:.4f}  "
          f"sign_correct={q2['sign_correct_rate']:.3f}  "
          f"saturation={q2['saturation_rate']:.3f}  pearson_cp_v={q2['pearson_cp_v']}",
          flush=True)
    q3 = summary["Q3_mcts_choice"]
    print(f"  Q3 MCTS choice: chosen_in_policy_top1_set={q3['chosen_in_v12_policy_top1_set_rate']:.3f}  "
          f"chosen_disagrees_pika={q3['chosen_disagrees_with_pika_best_rate']:.3f}",
          flush=True)
    pe = summary["policy_entropy_nats"]
    print(f"  policy entropy nats (p10/25/50/75/90): {pe.get(0.1):.2f}/{pe.get(0.25):.2f}/"
          f"{pe.get(0.5):.2f}/{pe.get(0.75):.2f}/{pe.get(0.9):.2f}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
