"""Oracle policy labeler — post-process shards to add a calibrated policy target.

Why
---
v10's training shards still use the MCTS visit distribution at sims=800 as the
policy target.  This signal is noisy for two reasons that mirror the v10 value
target story:

1. **Granularity**: visit counts at sims=800 are spiky (Dirichlet-noisy) — they
   amplify whatever the network's policy prior already says, then sharpen it.
   For non-tactical positions, the visit distribution can be heavily weighted
   on a single move when 2-3 alternatives are roughly equal.
2. **Distribution quality**: the MCTS visit distribution is *the model's own
   recursion* — it cannot teach the model anything that the model can't already
   discover.  This is the AlphaZero canonical policy target, but it's a closed
   loop.

Fix
---
For each position in a shard, query a strong Pikafish (default depth=8,
multipv=5) and collect the top-K moves with their centipawn evals.  Build:

    oracle_policy[move_i] = softmax(eval_i / temperature)
    oracle_policy[move_other] = epsilon  (small uniform mass over remaining legal)

Write three new shard fields (CSR-style, mirroring policy_idxs/policy_probs):
- ``oracle_policy_offsets`` (B+1,)  int64  — boundaries
- ``oracle_policy_idxs``    (Σ K,)  int64  — internal move indices (from*90+to)
- ``oracle_policy_probs``   (Σ K,)  float32 — softmax-derived probabilities

Training will combine MCTS-visit policy and oracle policy with weight ``α``
(set in TrainingConfig).  Positions where oracle generation failed have an
empty slice (offsets[i+1] == offsets[i]) and fall back to MCTS-only.

Usage
-----
    python tools/oracle_policy_labeler.py \\
        --input-shard-dir /path/to/selfplay_run/train/ \\
        --output-shard-dir /path/to/selfplay_run/train/ \\
        --depth 8 --multipv 5 \\
        --workers 8

Cost
----
At depth=8 multipv=5 on Pikafish, expect ~0.3-1.0s per position (vs ~1-2s for
oracle_value at depth 15).  The bigger expense is multipv overhead — Pikafish
runs an extra search per ranked move.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from pikafish_pool import PikafishJob, PikafishPool, PikafishResult  # noqa: E402
from pikafish_opponent import uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import canonical_action  # noqa: E402

# Mate evals come back as ±20000 cp from pikafish_pool; cap before softmax to
# prevent numeric overflow and to make the resulting distribution sane.
_CP_CLAMP = 1500.0


def _build_oracle_distribution(
    multipv_moves: list[tuple[str, int]] | None,
    temperature_cp: float,
    stm_is_black: bool,
    legal_idxs: list[int] | None = None,
    epsilon: float = 0.0,
    adaptive_temperature: bool = False,
    min_temperature_cp: float = 35.0,
    max_temperature_cp: float = 200.0,
    soft_gap_cp: float = 30.0,
    hard_gap_cp: float = 180.0,
) -> tuple[list[int], list[float], float]:
    """Convert Pikafish multipv result -> (canonical move_indices, probs).

    CRITICAL: Pikafish operates on raw board state and emits raw UCI moves. The
    rest of the training stack (state tensor, policy_idxs, legal_idxs) uses the
    CANONICAL frame (board mirrored when stm is black). We must apply the same
    canonical_action() transform here, or oracle_policy targets will reference
    different action indices than the model's policy_logits.

    BUG HISTORY: This function used to skip the canonicalization. v10/v11
    silently trained with mismatched oracle_policy targets for half the
    positions (black-to-move). The bug was harmless at training time because
    log_softmax was unmasked — wrong oracle indices just trained the wrong
    logits towards a soft target, but loss never went infinite. v12 added
    legal-mask softmax — wrong oracle indices land on legal_mask=False
    positions whose logits become -1e9, log_softmax = -∞, CE = +∞, gradient
    explosion, training catastrophically destroyed in 5 cycles (probe collapsed
    from 90% to 10%). Discovered 2026-04-30.

    Returns (indices, probs, selected_temperature_cp). Indices/probs are empty
    if the multipv result is missing or all moves are invalid.
    """
    if not multipv_moves:
        return [], [], float(temperature_cp)

    indices: list[int] = []
    capped_evals: list[float] = []
    legal_set = None if not legal_idxs else {int(x) for x in legal_idxs}
    for uci_move, cp in multipv_moves:
        try:
            raw_move = uci_move_to_internal(uci_move)
            canonical_idx = int(canonical_action(int(raw_move), bool(stm_is_black)))
        except Exception:
            continue
        if legal_set is not None and canonical_idx not in legal_set:
            # Dirty shard or protocol mismatch.  Do not emit an oracle target that
            # legal-masked CE would treat as impossible.
            continue
        capped = max(-_CP_CLAMP, min(_CP_CLAMP, float(cp)))
        indices.append(canonical_idx)
        capped_evals.append(capped)

    if not indices:
        return [], [], float(temperature_cp)

    selected_temperature = _select_temperature_cp(
        capped_evals=capped_evals,
        base_temperature_cp=float(temperature_cp),
        adaptive=bool(adaptive_temperature),
        min_temperature_cp=float(min_temperature_cp),
        max_temperature_cp=float(max_temperature_cp),
        soft_gap_cp=float(soft_gap_cp),
        hard_gap_cp=float(hard_gap_cp),
    )

    # Softmax over evals/temperature.  Eval is from side-to-move's POV so a
    # higher-eval move should get higher prob — that's already the orientation
    # Pikafish multipv emits at the root.
    max_eval = max(capped_evals)
    exps = [math.exp((e - max_eval) / max(1.0, selected_temperature)) for e in capped_evals]
    total = sum(exps)
    if total <= 0.0:
        # Degenerate: fall back to uniform over the K returned moves.
        probs = [1.0 / len(indices)] * len(indices)
    else:
        probs = [x / total for x in exps]

    if epsilon > 0.0:
        eps = min(max(float(epsilon), 0.0), 0.5)
        if legal_set:
            prob_map = {int(idx): (1.0 - eps) * float(prob) for idx, prob in zip(indices, probs)}
            legal_mass = eps / float(len(legal_set))
            for idx in legal_set:
                prob_map[int(idx)] = prob_map.get(int(idx), 0.0) + legal_mass
            indices = sorted(prob_map)
            probs = [prob_map[idx] for idx in indices]
        else:
            # Backward-compatible fallback for old shards without legal_idxs:
            # smooth over returned MultiPV moves only.
            probs = [(1.0 - eps) * p + eps / len(probs) for p in probs]
        s = sum(probs)
        probs = [p / s for p in probs] if s > 0 else [1.0 / len(probs)] * len(probs)

    return indices, probs, selected_temperature


def _select_temperature_cp(
    *,
    capped_evals: list[float],
    base_temperature_cp: float,
    adaptive: bool,
    min_temperature_cp: float,
    max_temperature_cp: float,
    soft_gap_cp: float,
    hard_gap_cp: float,
) -> float:
    if not adaptive or len(capped_evals) < 2:
        return max(1.0, float(base_temperature_cp))
    ordered = sorted(capped_evals, reverse=True)
    gap = max(0.0, float(ordered[0] - ordered[1]))
    lo = min(float(soft_gap_cp), float(hard_gap_cp))
    hi = max(float(soft_gap_cp), float(hard_gap_cp))
    if gap <= lo:
        return max(1.0, float(max_temperature_cp))
    if gap >= hi:
        return max(1.0, float(min_temperature_cp))
    t = (gap - lo) / max(1e-6, hi - lo)
    return max(1.0, float(max_temperature_cp) + t * (float(min_temperature_cp) - float(max_temperature_cp)))


def label_one_shard(
    shard_path: Path,
    output_path: Path,
    pool: PikafishPool,
    depth: int,
    multipv: int,
    temperature_cp: float,
    adaptive_temperature: bool,
    min_temperature_cp: float,
    max_temperature_cp: float,
    soft_gap_cp: float,
    hard_gap_cp: float,
    legal_smoothing: float,
    max_wait_per_shard_s: float | None = None,
    skip_if_already_labeled: bool = True,
) -> dict:
    """Read a shard, query Pikafish multipv for each FEN, write oracle_policy.

    Returns stats dict {samples, labeled, errors, duration_s, skipped}.
    """
    t0 = time.monotonic()
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    n_samples = int(payload["state"].shape[0])

    if "fens" not in payload:
        return {"samples": n_samples, "labeled": 0, "errors": 0,
                "duration_s": 0.0, "skipped": True,
                "reason": "shard has no 'fens' field (old format)"}

    fens = list(payload["fens"])
    if len(fens) != n_samples:
        return {"samples": n_samples, "labeled": 0, "errors": 0,
                "duration_s": 0.0, "skipped": True,
                "reason": f"len(fens)={len(fens)} != n_samples={n_samples}"}

    # CRITICAL (v12 bug fix): need stm_is_black per position to canonicalize Pikafish
    # raw moves. Without canonicalization, oracle_policy_idxs reference action indices
    # in the unmirrored frame while everything else (state tensor, policy_idxs,
    # legal_idxs) is in the canonical frame — half the positions (black-to-move)
    # store wrong indices. See _build_oracle_distribution docstring for full history.
    stm_is_black_tensor = payload.get("stm_is_black")
    if stm_is_black_tensor is None:
        return {"samples": n_samples, "labeled": 0, "errors": 0,
                "duration_s": 0.0, "skipped": True,
                "reason": "shard missing 'stm_is_black' field — cannot canonicalize moves"}
    legal_offsets = payload.get("legal_offsets")
    legal_idxs = payload.get("legal_idxs")
    if legal_offsets is not None:
        legal_offsets = legal_offsets.to(torch.int64)
    if legal_idxs is not None:
        legal_idxs = legal_idxs.to(torch.int64)

    if skip_if_already_labeled and "oracle_policy_offsets" in payload:
        existing = payload["oracle_policy_offsets"]
        if isinstance(existing, torch.Tensor) and existing.numel() == n_samples + 1:
            meta = payload.get("oracle_policy_meta") or {}
            if isinstance(meta, dict) and bool(meta.get("canonical_action", False)):
                return {"samples": n_samples, "labeled": 0, "errors": 0,
                        "duration_s": 0.0, "skipped": True,
                        "reason": "already labeled with canonical_action meta"}
            # v12.5 safety: old labels without canonical_action=True are treated
            # as suspect and re-labeled even when --skip-already-labeled is set.

    # Build jobs.  Index = position-in-shard.  Skip empty/missing FENs.
    jobs: list[PikafishJob] = []
    for i, fen in enumerate(fens):
        if not fen or not isinstance(fen, str):
            continue
        jobs.append(PikafishJob(
            index=i, fen=fen,
            depth=int(depth),
            multipv=int(max(1, multipv)),
        ))

    if not jobs:
        return {"samples": n_samples, "labeled": 0, "errors": 0,
                "duration_s": time.monotonic() - t0, "skipped": True,
                "reason": "no valid FENs in shard"}

    pool.submit_all(jobs)
    raw_results = pool.collect(len(jobs), timeout_s=max_wait_per_shard_s)
    by_index: dict[int, PikafishResult] = {r.index: r for r in raw_results}

    # Assemble CSR oracle policy.  Empty slice for failed positions.
    offsets: list[int] = [0]
    flat_idxs: list[int] = []
    flat_probs: list[float] = []
    used_temperatures: list[float] = []
    n_labeled = 0
    n_errors = 0
    for i in range(n_samples):
        r = by_index.get(i)
        if r is None or r.error:
            n_errors += 1
            offsets.append(offsets[-1])
            continue
        moves_for_pos = r.multipv_moves
        if not moves_for_pos and r.best_move and r.best_move != "0000":
            # multipv=1 case or single result — synthesize 1-move slot
            moves_for_pos = [(r.best_move, int(r.eval_cp))]
        legal_i: list[int] | None = None
        if legal_offsets is not None and legal_idxs is not None:
            lg_s = int(legal_offsets[i].item())
            lg_e = int(legal_offsets[i + 1].item())
            if lg_e > lg_s:
                legal_i = [int(x) for x in legal_idxs[lg_s:lg_e].tolist()]
        idxs_i, probs_i, temp_i = _build_oracle_distribution(
            multipv_moves=moves_for_pos,
            temperature_cp=temperature_cp,
            stm_is_black=bool(stm_is_black_tensor[i].item()),
            legal_idxs=legal_i,
            epsilon=float(legal_smoothing),
            adaptive_temperature=bool(adaptive_temperature),
            min_temperature_cp=float(min_temperature_cp),
            max_temperature_cp=float(max_temperature_cp),
            soft_gap_cp=float(soft_gap_cp),
            hard_gap_cp=float(hard_gap_cp),
        )
        if not idxs_i:
            n_errors += 1
            offsets.append(offsets[-1])
            continue
        flat_idxs.extend(idxs_i)
        flat_probs.extend(probs_i)
        used_temperatures.append(float(temp_i))
        offsets.append(offsets[-1] + len(idxs_i))
        n_labeled += 1

    # Atomic write
    payload["oracle_policy_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    payload["oracle_policy_idxs"] = torch.tensor(flat_idxs, dtype=torch.int64)
    payload["oracle_policy_probs"] = torch.tensor(flat_probs, dtype=torch.float32)
    payload["oracle_policy_meta"] = {
        "depth": int(depth),
        "multipv": int(multipv),
        "temperature_cp": float(temperature_cp),
        "adaptive_temperature": bool(adaptive_temperature),
        "min_temperature_cp": float(min_temperature_cp),
        "max_temperature_cp": float(max_temperature_cp),
        "soft_gap_cp": float(soft_gap_cp),
        "hard_gap_cp": float(hard_gap_cp),
        "legal_smoothing": float(legal_smoothing),
        "canonical_action": True,
        "temperature_observed_min": min(used_temperatures) if used_temperatures else None,
        "temperature_observed_max": max(used_temperatures) if used_temperatures else None,
        "temperature_observed_mean": (
            sum(used_temperatures) / len(used_temperatures) if used_temperatures else None
        ),
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)

    return {
        "samples": n_samples,
        "labeled": n_labeled,
        "errors": n_errors,
        "duration_s": time.monotonic() - t0,
        "skipped": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-shard-dir", required=True)
    p.add_argument("--output-shard-dir", default=None,
                   help="If omitted, overwrites in place.")
    p.add_argument("--depth", type=int, default=8,
                   help="Pikafish depth for multipv eval. Default 8 (lower than oracle_value's "
                        "depth=15 — multipv adds K-fold per-move work, depth=8 keeps per-position "
                        "cost in the 0.3-1.0s range).")
    p.add_argument("--multipv", type=int, default=5,
                   help="Top-K moves to extract per position. Default 5.")
    p.add_argument("--temperature-cp", type=float, default=50.0,
                   help="Softmax temperature for cp evals. 50 cp = factor of e per "
                        "50cp difference (sharp, near-argmax). Lower = sharper, higher = smoother. "
                        "v11 used 200 (smooth); v12 default 50 to push policy distillation closer "
                        "to imitation of Pikafish best-move (other Agent feedback: action-value "
                        "distillation should sharpen around the best move).")
    p.add_argument("--adaptive-temperature", action=argparse.BooleanOptionalAction, default=False,
                   help="v12.5: choose a per-position temperature from the MultiPV top gap. "
                        "Small top1-top2 gap => smooth target; large gap => sharp target.")
    p.add_argument("--min-temperature-cp", type=float, default=35.0,
                   help="Sharpest adaptive temperature when top1-top2 gap is large. Default 35.")
    p.add_argument("--max-temperature-cp", type=float, default=200.0,
                   help="Smoothest adaptive temperature when top1-top2 gap is small. Default 200.")
    p.add_argument("--soft-gap-cp", type=float, default=30.0,
                   help="Top1-top2 gap at/below which adaptive temp uses --max-temperature-cp.")
    p.add_argument("--hard-gap-cp", type=float, default=180.0,
                   help="Top1-top2 gap at/above which adaptive temp uses --min-temperature-cp.")
    p.add_argument("--legal-smoothing", type=float, default=0.02,
                   help="v12.5: redistribute this probability mass over all legal moves when "
                        "legal_idxs are available. Default 0.02. Set 0 for old top-K-only targets.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--shard-glob", default="shard_*.pt")
    p.add_argument("--max-wait-per-shard-s", type=float, default=1800.0)
    p.add_argument("--skip-already-labeled", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    in_dir = Path(args.input_shard_dir)
    out_dir = Path(args.output_shard_dir) if args.output_shard_dir else in_dir
    if not in_dir.is_dir():
        raise SystemExit(f"--input-shard-dir not found: {in_dir}")

    shards = sorted(in_dir.glob(args.shard_glob))
    if not shards:
        raise SystemExit(f"no shards matching {args.shard_glob!r} under {in_dir}")
    print(f"found {len(shards)} shard(s) under {in_dir}", flush=True)

    pool = PikafishPool(
        num_workers=int(args.workers),
        binary_path=args.pikafish_binary,
        threads_per_worker=int(args.threads_per_worker),
        hash_mb=int(args.hash_mb),
    )
    print(f"started PikafishPool: {args.workers} workers × {args.threads_per_worker} threads, "
          f"hash={args.hash_mb}MB, depth={args.depth}, multipv={args.multipv}", flush=True)

    total_samples = 0
    total_labeled = 0
    total_errors = 0
    total_skipped = 0
    t_start = time.monotonic()
    try:
        for i, shard_path in enumerate(shards):
            out_path = out_dir / shard_path.name
            stats = label_one_shard(
                shard_path=shard_path,
                output_path=out_path,
                pool=pool,
                depth=int(args.depth),
                multipv=int(args.multipv),
                temperature_cp=float(args.temperature_cp),
                adaptive_temperature=bool(args.adaptive_temperature),
                min_temperature_cp=float(args.min_temperature_cp),
                max_temperature_cp=float(args.max_temperature_cp),
                soft_gap_cp=float(args.soft_gap_cp),
                hard_gap_cp=float(args.hard_gap_cp),
                legal_smoothing=float(args.legal_smoothing),
                max_wait_per_shard_s=float(args.max_wait_per_shard_s),
                skip_if_already_labeled=bool(args.skip_already_labeled),
            )
            total_samples += stats["samples"]
            total_labeled += stats["labeled"]
            total_errors += stats["errors"]
            if stats["skipped"]:
                total_skipped += 1
                print(f"  shard {i+1}/{len(shards)} {shard_path.name}: SKIPPED "
                      f"({stats.get('reason', '')})", flush=True)
            else:
                rate = stats["labeled"] / max(stats["duration_s"], 1e-6)
                print(f"  shard {i+1}/{len(shards)} {shard_path.name}: "
                      f"labeled={stats['labeled']}/{stats['samples']} "
                      f"errors={stats['errors']} dt={stats['duration_s']:.0f}s "
                      f"({rate:.1f} pos/s)", flush=True)
    finally:
        pool.close()

    dt_total = time.monotonic() - t_start
    print()
    print(f"DONE: {total_labeled}/{total_samples} positions policy-labeled "
          f"({total_errors} errors, {total_skipped} shards skipped) in {dt_total:.0f}s",
          flush=True)
    if total_labeled > 0:
        print(f"  effective throughput: {total_labeled/dt_total:.1f} pos/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
