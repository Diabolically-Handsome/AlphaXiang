"""Oracle value labeler — post-process shards to add a calibrated value target.

Why
---
v7's training shards use ``z`` (game outcome ∈ {-1, 0, +1}) as the value-head
target.  This signal is extremely noisy for two reasons:

1. **Granularity**: a position at ply 30 of a 100-ply game gets the same z as
   the final position, even though the position-level value should differ wildly.
2. **Distribution shift**: z is calibrated for the training-time opponent
   (Pikafish d=2..d=5).  When v7 plays CNN at deployment, its value head is
   miscalibrated — the **OOD over-search trap** documented in our v7-vs-CNN
   experiment (88% at 800 sims → 67% at 1600 sims, regression).

Fix
---
For each position in a shard, query a *strong* Pikafish (default depth=15) on
the position's FEN, extract the centipawn eval, and write
``oracle_value = tanh(cp / 500)`` as a new shard field.  Training will use this
calibrated value when present, falling back to the noisy ``z * value_target_scale``
otherwise.

Usage
-----
    python tools/oracle_value_labeler.py \\
        --input-shard-dir /path/to/selfplay_run/train/ \\
        --output-shard-dir /path/to/selfplay_run/train/ \\
        --depth 15 \\
        --workers 8

When ``--input-shard-dir == --output-shard-dir`` the tool overwrites in place
(after writing each shard atomically via ``shard.tmp`` rename, so partial
labeling never corrupts data).

Error handling
--------------
* Shards without a ``fens`` field (older format) are skipped with a warning.
* Positions where Pikafish errors out get ``oracle_value = NaN`` written.  The
  training side filters NaN samples out of the value-loss numerator (via a
  mask), so they fall back to z-loss for those positions only.
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

from pikafish_pool import (  # noqa: E402
    PikafishJob,
    PikafishPool,
    PikafishPoolTimeout,
    PikafishResult,
)


def _cp_to_tanh(cp: int, scale: float = 500.0) -> float:
    """Map centipawn eval to [-1, +1] via tanh(cp/scale).

    scale=500 matches distillation_generator's existing convention.  Pikafish
    cp is from side-to-move's POV, which is what the model's value head expects.
    """
    return float(math.tanh(float(cp) / float(scale)))


def label_one_shard(
    shard_path: Path,
    output_path: Path,
    pool: PikafishPool,
    depth: int,
    cp_to_tanh_scale: float,
    max_wait_per_shard_s: float | None = None,
    skip_if_already_labeled: bool = True,
) -> dict:
    """Read a shard, query Pikafish for each FEN, write shard+oracle_value.

    Returns a stats dict {samples, labeled, errors, duration_s, skipped}.
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

    existing_oracle: torch.Tensor | None = None
    existing_depth = None
    if isinstance(payload.get("oracle_meta"), dict):
        existing_depth = payload["oracle_meta"].get("depth")
    if skip_if_already_labeled and "oracle_value" in payload:
        existing = payload["oracle_value"]
        if isinstance(existing, torch.Tensor) and existing.shape[0] == n_samples:
            n_nan = int(torch.isnan(existing).sum().item())
            if n_nan == 0 and existing_depth == int(depth):
                return {"samples": n_samples, "labeled": 0, "errors": 0,
                        "duration_s": 0.0, "skipped": True,
                        "reason": "already fully labeled"}
            if existing_depth == int(depth):
                existing_oracle = existing.detach().cpu().to(torch.float32).clone()

    oracle = (
        existing_oracle
        if existing_oracle is not None
        else torch.full((n_samples,), float("nan"), dtype=torch.float32)
    )

    # Build jobs.  Index = position-in-shard.  Skip empty/missing FENs.
    valid_indices: list[int] = []
    jobs: list[PikafishJob] = []
    for i, fen in enumerate(fens):
        if not fen or not isinstance(fen, str):
            continue
        if not bool(torch.isnan(oracle[i]).item()):
            continue
        valid_indices.append(i)
        jobs.append(PikafishJob(index=i, fen=fen, depth=int(depth)))

    if not jobs:
        n_labeled = int(torch.isfinite(oracle).sum().item())
        if n_labeled > 0:
            return {"samples": n_samples, "labeled": n_labeled,
                    "new_labeled": 0, "errors": n_samples - n_labeled,
                    "duration_s": time.monotonic() - t0, "skipped": True,
                    "reason": "no remaining valid FENs to label",
                    "timed_out": False}
        return {"samples": n_samples, "labeled": 0, "new_labeled": 0, "errors": 0,
                "duration_s": time.monotonic() - t0, "skipped": True,
                "reason": "no valid FENs in shard", "timed_out": False}

    # Submit + collect
    pool.submit_all(jobs)
    timed_out = False
    try:
        raw_results = pool.collect(len(jobs), timeout_s=max_wait_per_shard_s)
    except PikafishPoolTimeout as exc:
        raw_results = exc.partial_results
        timed_out = True
    by_index: dict[int, PikafishResult] = {r.index: r for r in raw_results}

    # Build oracle_value tensor (B,) with NaN for missing/error positions
    n_new_labeled = 0
    for i in valid_indices:
        r = by_index.get(i)
        if r is None or r.error:
            continue
        oracle[i] = _cp_to_tanh(int(r.eval_cp), scale=cp_to_tanh_scale)
        n_new_labeled += 1
    n_labeled = int(torch.isfinite(oracle).sum().item())
    n_errors = n_samples - n_labeled

    # Atomically write new payload.  Save with a .tmp suffix then rename so a
    # crash mid-write never corrupts the shard.
    payload["oracle_value"] = oracle
    payload["oracle_meta"] = {
        "depth": int(depth),
        "cp_to_tanh_scale": float(cp_to_tanh_scale),
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timed_out": bool(timed_out),
        "timeout_s": None if max_wait_per_shard_s is None else float(max_wait_per_shard_s),
        "missing": int(n_errors),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)

    return {
        "samples": n_samples,
        "labeled": n_labeled,
        "new_labeled": n_new_labeled,
        "errors": n_errors,
        "duration_s": time.monotonic() - t0,
        "skipped": False,
        "timed_out": timed_out,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-shard-dir", required=True,
                   help="Directory containing shard_*.pt files to label")
    p.add_argument("--output-shard-dir", default=None,
                   help="Where to write labeled shards. If omitted, overwrites in place.")
    p.add_argument("--depth", type=int, default=15,
                   help="Pikafish depth for oracle eval. Default 15 (~1-2s/position, "
                        "much stronger than the d=6 used in distillation).")
    p.add_argument("--cp-to-tanh-scale", type=float, default=500.0,
                   help="Centipawn-to-tanh scaling factor. tanh(cp/scale) ∈ [-1, +1]. "
                        "Default 500 matches distillation_generator convention.")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of Pikafish processes in parallel.")
    p.add_argument("--threads-per-worker", type=int, default=1,
                   help="Threads per Pikafish process (multiply with --workers for total CPU).")
    p.add_argument("--hash-mb", type=int, default=64,
                   help="Pikafish hash table size per worker. Larger = better at d=15. Default 64.")
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--shard-glob", default="shard_*.pt",
                   help="Glob pattern under --input-shard-dir.")
    p.add_argument("--max-wait-per-shard-s", type=float, default=1800.0,
                   help="Per-shard timeout. Long because d=15 × 1024 positions can take a while.")
    p.add_argument("--skip-already-labeled", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip shards that already have a complete oracle_value field "
                        "(useful for resuming partial runs). Pass --no-skip-already-labeled "
                        "to force re-label.")
    args = p.parse_args()

    in_dir = Path(args.input_shard_dir)
    out_dir = Path(args.output_shard_dir) if args.output_shard_dir else in_dir
    if not in_dir.is_dir():
        raise SystemExit(f"--input-shard-dir not found: {in_dir}")

    shards = sorted(in_dir.glob(args.shard_glob))
    if not shards:
        raise SystemExit(f"no shards matching {args.shard_glob!r} under {in_dir}")
    print(f"found {len(shards)} shard(s) under {in_dir}", flush=True)

    def new_pool() -> PikafishPool:
        pool_obj = PikafishPool(
            num_workers=int(args.workers),
            binary_path=args.pikafish_binary,
            threads_per_worker=int(args.threads_per_worker),
            hash_mb=int(args.hash_mb),
        )
        print(f"started PikafishPool: {args.workers} workers × {args.threads_per_worker} threads, "
              f"hash={args.hash_mb}MB, depth={args.depth}", flush=True)
        return pool_obj

    pool = new_pool()

    total_samples = 0
    total_labeled = 0
    total_errors = 0
    total_skipped = 0
    fatal_timeout = False
    t_start = time.monotonic()
    try:
        for i, shard_path in enumerate(shards):
            out_path = out_dir / shard_path.name
            stats = label_one_shard(
                shard_path=shard_path,
                output_path=out_path,
                pool=pool,
                depth=int(args.depth),
                cp_to_tanh_scale=float(args.cp_to_tanh_scale),
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
                timeout_note = " TIMED_OUT_PARTIAL" if stats.get("timed_out") else ""
                print(f"  shard {i+1}/{len(shards)} {shard_path.name}: "
                      f"labeled={stats['labeled']}/{stats['samples']} "
                      f"new={stats.get('new_labeled', stats['labeled'])} "
                      f"errors={stats['errors']} dt={stats['duration_s']:.0f}s "
                      f"({rate:.1f} pos/s){timeout_note}", flush=True)
                if stats.get("timed_out"):
                    print("    timeout reached; partial labels were written and "
                          "remaining positions are NaN fallback targets", flush=True)
                    if int(stats.get("labeled", 0)) == 0:
                        fatal_timeout = True
                    pool.close()
                    pool = new_pool()
    finally:
        pool.close()

    dt_total = time.monotonic() - t_start
    print()
    print(f"DONE: {total_labeled}/{total_samples} positions labeled "
          f"({total_errors} errors, {total_skipped} shards skipped) in {dt_total:.0f}s",
          flush=True)
    if total_labeled > 0:
        print(f"  effective throughput: {total_labeled/dt_total:.1f} pos/s",
              flush=True)
    if fatal_timeout:
        print("ERROR: at least one shard timed out before producing any labels", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
