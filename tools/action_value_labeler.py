"""Label candidate actions with teacher Q-values for v12.5 distillation.

For each selected position in a shard, build a small candidate move set
(oracle-policy top-K, MCTS target top-K, chosen move), push each move, ask
Pikafish to evaluate the child position, and store:

    teacher_q_offsets: (N+1,) int64
    teacher_q_idxs:    (sum K_i,) int64 canonical action indices
    teacher_q_values:  (sum K_i,) float32 centipawns from root STM perspective

Training can then add --teacher-q-loss-weight > 0 to distill a listwise
action-value target without changing the model architecture.
"""
from __future__ import annotations

import argparse
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

from pikafish_pool import PikafishJob, PikafishPool, PikafishResult  # noqa: E402
from pikafish_opponent import uci_move_to_internal  # noqa: E402
from xiangqi_mcts_ext import Board, canonical_action  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402

TERMINAL_ONGOING = -1
_CP_CLAMP = 20000.0


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


def _top_from_csr(
    payload: dict[str, Any],
    prefix: str,
    i: int,
    top_k: int,
) -> list[int]:
    offsets = payload.get(f"{prefix}_offsets")
    idxs = payload.get(f"{prefix}_idxs")
    probs = payload.get(f"{prefix}_probs")
    if offsets is None or idxs is None or probs is None or top_k <= 0:
        return []
    offsets = offsets.to(torch.int64)
    idxs = idxs.to(torch.int64)
    probs = probs.float()
    start = int(offsets[i].item())
    end = int(offsets[i + 1].item())
    if end <= start:
        return []
    row_idxs = idxs[start:end]
    row_probs = probs[start:end]
    order = torch.argsort(row_probs, descending=True)
    return [int(row_idxs[j].item()) for j in order[:top_k]]


def _legal_set(payload: dict[str, Any], i: int) -> set[int]:
    offsets = payload.get("legal_offsets")
    idxs = payload.get("legal_idxs")
    if offsets is None or idxs is None:
        return set()
    offsets = offsets.to(torch.int64)
    idxs = idxs.to(torch.int64)
    start = int(offsets[i].item())
    end = int(offsets[i + 1].item())
    if end <= start:
        return set()
    return {int(x) for x in idxs[start:end].tolist()}


def _candidate_actions(
    payload: dict[str, Any],
    i: int,
    *,
    model_topk: list[int] | None,
    oracle_top_k: int,
    mcts_top_k: int,
    include_chosen: bool,
    max_candidates: int,
) -> list[int]:
    out: list[int] = []
    for idx in model_topk or []:
        if idx not in out:
            out.append(int(idx))
    for idx in _top_from_csr(payload, "oracle_policy", i, oracle_top_k):
        if idx not in out:
            out.append(idx)
    for idx in _top_from_csr(payload, "policy", i, mcts_top_k):
        if idx not in out:
            out.append(idx)
    if include_chosen and "chosen_move" in payload:
        chosen = int(payload["chosen_move"][i].item())
        if chosen not in out:
            out.append(chosen)
    if "bad_move" in payload:
        bad_move = int(payload["bad_move"][i].item())
        if bad_move >= 0 and bad_move not in out:
            out.append(bad_move)
    return out[:max(1, int(max_candidates))]


@torch.no_grad()
def _model_topk_candidates(
    *,
    payload: dict[str, Any],
    model: torch.nn.Module | None,
    device: torch.device,
    top_k: int,
    batch_size: int,
    use_bfloat16: bool,
) -> list[list[int]] | None:
    if model is None or top_k <= 0:
        return None
    state = payload.get("state")
    if not isinstance(state, torch.Tensor):
        return None
    n = int(state.shape[0])
    legal_offsets = payload.get("legal_offsets")
    legal_idxs = payload.get("legal_idxs")
    if not isinstance(legal_offsets, torch.Tensor) or not isinstance(legal_idxs, torch.Tensor):
        return [[] for _ in range(n)]

    legal_offsets = legal_offsets.to(torch.int64)
    legal_idxs = legal_idxs.to(torch.int64)
    out: list[list[int]] = [[] for _ in range(n)]
    model.eval()
    state = state.to(torch.float32)
    autocast_enabled = bool(use_bfloat16 and device.type == "cuda")
    for start in range(0, n, max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), n)
        batch = state[start:stop].to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(batch)["policy_logits"].float().cpu()
        for local_i, row_i in enumerate(range(start, stop)):
            leg_start = int(legal_offsets[row_i].item())
            leg_end = int(legal_offsets[row_i + 1].item())
            if leg_end <= leg_start:
                continue
            row_legal = legal_idxs[leg_start:leg_end]
            row_logits = logits[local_i, row_legal]
            k = min(int(top_k), int(row_legal.numel()))
            if k <= 0:
                continue
            order = torch.topk(row_logits, k=k).indices
            out[row_i] = [int(row_legal[j].item()) for j in order]
    return out


def _terminal_q_cp(board: Board, root_stm_black: bool, *, max_plies: int,
                   repeat_limit: int, repeat_min_ply: int, no_capture_limit: int) -> float | None:
    term = int(board.terminal_code(max_plies, repeat_limit, repeat_min_ply, no_capture_limit))
    if term == TERMINAL_ONGOING:
        return None
    red_result = int(board.terminal_result_red_view(term))
    if red_result == 0:
        return 0.0
    root_is_red = not bool(root_stm_black)
    root_won = (red_result > 0) == root_is_red
    return 20000.0 if root_won else -20000.0


def label_one_shard(
    *,
    shard_path: Path,
    output_path: Path,
    pool: PikafishPool,
    depth: int,
    oracle_top_k: int,
    mcts_top_k: int,
    max_candidates: int,
    only_hard: bool,
    min_sample_weight: float,
    include_chosen: bool,
    candidate_model: torch.nn.Module | None,
    candidate_model_device: torch.device,
    model_top_k: int,
    model_batch_size: int,
    model_use_bfloat16: bool,
    skip_if_already_labeled: bool,
    max_wait_per_shard_s: float | None,
    max_plies: int,
    repeat_limit: int,
    repeat_min_ply: int,
    no_capture_limit: int,
) -> dict[str, Any]:
    t0 = time.monotonic()
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    n = int(payload["state"].shape[0])
    if skip_if_already_labeled and "teacher_q_offsets" in payload:
        existing = payload["teacher_q_offsets"]
        if isinstance(existing, torch.Tensor) and int(existing.numel()) == n + 1:
            return {"samples": n, "labeled_rows": 0, "entries": 0, "skipped": True,
                    "reason": "already labeled", "duration_s": 0.0}
    fens = payload.get("fens")
    stm = payload.get("stm_is_black")
    if not isinstance(fens, list) or len(fens) != n:
        return {"samples": n, "labeled_rows": 0, "entries": 0, "skipped": True,
                "reason": "missing fens", "duration_s": 0.0}
    if not isinstance(stm, torch.Tensor) or int(stm.numel()) != n:
        return {"samples": n, "labeled_rows": 0, "entries": 0, "skipped": True,
                "reason": "missing stm_is_black", "duration_s": 0.0}

    sample_weight = payload.get("sample_weight")
    model_topk_by_row = _model_topk_candidates(
        payload=payload,
        model=candidate_model,
        device=candidate_model_device,
        top_k=int(model_top_k),
        batch_size=int(model_batch_size),
        use_bfloat16=bool(model_use_bfloat16),
    )
    q_by_row: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    jobs: list[PikafishJob] = []
    job_map: list[tuple[int, int]] = []  # (row, canonical_idx)
    skipped_rows = 0
    illegal_candidates = 0

    for i in range(n):
        if only_hard:
            if sample_weight is None or float(sample_weight[i].item()) < float(min_sample_weight):
                skipped_rows += 1
                continue
        fen = fens[i]
        if not fen:
            skipped_rows += 1
            continue
        root_stm_black = bool(stm[i].item())
        legal = _legal_set(payload, i)
        row_model_topk = None if model_topk_by_row is None else model_topk_by_row[i]
        candidates = _candidate_actions(
            payload,
            i,
            model_topk=row_model_topk,
            oracle_top_k=int(oracle_top_k),
            mcts_top_k=int(mcts_top_k),
            include_chosen=bool(include_chosen),
            max_candidates=int(max_candidates),
        )
        survival_played = payload.get("survival_played_move_uci")
        if isinstance(survival_played, list) and i < len(survival_played):
            try:
                played_raw = int(uci_move_to_internal(str(survival_played[i])[:4]))
                played_canonical = int(canonical_action(played_raw, root_stm_black))
                if played_canonical not in candidates:
                    candidates.append(played_canonical)
            except Exception:
                pass
        if legal:
            candidates = [idx for idx in candidates if int(idx) in legal]
        if not candidates:
            skipped_rows += 1
            continue

        for canonical_idx in candidates:
            board = Board()
            board.set_fen(_pad_fen(str(fen)))
            raw_move = int(canonical_action(int(canonical_idx), root_stm_black))
            if not bool(board.is_legal(raw_move)):
                illegal_candidates += 1
                continue
            board.push_legal(raw_move)
            terminal_q = _terminal_q_cp(
                board,
                root_stm_black,
                max_plies=max_plies,
                repeat_limit=repeat_limit,
                repeat_min_ply=repeat_min_ply,
                no_capture_limit=no_capture_limit,
            )
            if terminal_q is not None:
                q_by_row[i].append((int(canonical_idx), float(terminal_q)))
                continue
            jobs.append(PikafishJob(
                index=len(job_map),
                fen=_pad_fen(board.fen()),
                depth=int(depth),
                multipv=1,
            ))
            job_map.append((i, int(canonical_idx)))

    if jobs:
        pool.submit_all(jobs)
        results = pool.collect(len(jobs), timeout_s=max_wait_per_shard_s)
        by_index: dict[int, PikafishResult] = {r.index: r for r in results}
        for job_idx, (row, canonical_idx) in enumerate(job_map):
            r = by_index.get(job_idx)
            if r is None or r.error:
                continue
            # Child eval is from the child side-to-move perspective; after our
            # candidate move that is the opponent, so negate to root STM POV.
            q_cp = -float(r.eval_cp)
            q_cp = max(-_CP_CLAMP, min(_CP_CLAMP, q_cp))
            q_by_row[row].append((int(canonical_idx), q_cp))

    offsets = [0]
    flat_idxs: list[int] = []
    flat_values: list[float] = []
    labeled_rows = 0
    for row in q_by_row:
        if row:
            # Deduplicate, keeping the first value for each candidate.
            dedup: dict[int, float] = {}
            for idx, q in row:
                dedup.setdefault(int(idx), float(q))
            for idx, q in dedup.items():
                flat_idxs.append(idx)
                flat_values.append(q)
            labeled_rows += 1
        offsets.append(len(flat_idxs))

    payload["teacher_q_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    payload["teacher_q_idxs"] = torch.tensor(flat_idxs, dtype=torch.int64)
    payload["teacher_q_values"] = torch.tensor(flat_values, dtype=torch.float32)
    payload["teacher_q_meta"] = {
        "depth": int(depth),
        "value_unit": "cp_root_pov",
        "oracle_top_k": int(oracle_top_k),
        "mcts_top_k": int(mcts_top_k),
        "model_top_k": int(model_top_k),
        "max_candidates": int(max_candidates),
        "only_hard": bool(only_hard),
        "min_sample_weight": float(min_sample_weight),
        "include_chosen": bool(include_chosen),
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    return {
        "samples": n,
        "labeled_rows": labeled_rows,
        "entries": len(flat_idxs),
        "jobs": len(jobs),
        "skipped_rows": skipped_rows,
        "illegal_candidates": illegal_candidates,
        "skipped": False,
        "duration_s": time.monotonic() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-shard-dir", required=True)
    p.add_argument("--output-shard-dir", default=None)
    p.add_argument("--shard-glob", default="shard_*.pt")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--hash-mb", type=int, default=64)
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--oracle-top-k", type=int, default=6)
    p.add_argument("--mcts-top-k", type=int, default=3)
    p.add_argument("--candidate-checkpoint", default=None,
                   help="Optional student checkpoint. If set, the model's top legal policy "
                        "moves are added to the candidate set before Pikafish refutation.")
    p.add_argument("--model-top-k", type=int, default=0,
                   help="Number of current-model top legal moves to include per position. "
                        "Requires --candidate-checkpoint. Default 0.")
    p.add_argument("--model-device", default="cuda:0")
    p.add_argument("--model-batch-size", type=int, default=128)
    p.add_argument("--disable-model-bf16", action="store_true")
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--only-hard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-sample-weight", type=float, default=2.0)
    p.add_argument("--include-chosen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-already-labeled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-wait-per-shard-s", type=float, default=3600.0)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--repeat-limit", type=int, default=6)
    p.add_argument("--repeat-min-ply", type=int, default=30)
    p.add_argument("--no-capture-limit", type=int, default=60)
    args = p.parse_args()

    in_dir = Path(args.input_shard_dir)
    out_dir = Path(args.output_shard_dir) if args.output_shard_dir else in_dir
    shards = sorted(in_dir.glob(args.shard_glob))
    if not shards:
        raise SystemExit(f"no shards matching {args.shard_glob!r} under {in_dir}")

    pool = PikafishPool(
        num_workers=int(args.workers),
        binary_path=args.pikafish_binary,
        threads_per_worker=int(args.threads_per_worker),
        hash_mb=int(args.hash_mb),
    )
    candidate_model = None
    candidate_model_device = torch.device(args.model_device)
    if args.candidate_checkpoint is not None and int(args.model_top_k) > 0:
        state = torch.load(Path(args.candidate_checkpoint), map_location="cpu", weights_only=False)
        candidate_model = build_model_from_checkpoint_state(state)
        candidate_model.to(candidate_model_device).eval()
        for parameter in candidate_model.parameters():
            parameter.requires_grad = False
        print(
            f"loaded candidate model from {args.candidate_checkpoint} on {candidate_model_device} "
            f"model_top_k={int(args.model_top_k)}",
            flush=True,
        )
    totals = {"samples": 0, "labeled_rows": 0, "entries": 0, "jobs": 0, "skipped_shards": 0}
    try:
        for idx, shard_path in enumerate(shards, start=1):
            stats = label_one_shard(
                shard_path=shard_path,
                output_path=out_dir / shard_path.name,
                pool=pool,
                depth=int(args.depth),
                oracle_top_k=int(args.oracle_top_k),
                mcts_top_k=int(args.mcts_top_k),
                max_candidates=int(args.max_candidates),
                only_hard=bool(args.only_hard),
                min_sample_weight=float(args.min_sample_weight),
                include_chosen=bool(args.include_chosen),
                candidate_model=candidate_model,
                candidate_model_device=candidate_model_device,
                model_top_k=int(args.model_top_k),
                model_batch_size=int(args.model_batch_size),
                model_use_bfloat16=not bool(args.disable_model_bf16),
                skip_if_already_labeled=bool(args.skip_already_labeled),
                max_wait_per_shard_s=float(args.max_wait_per_shard_s),
                max_plies=int(args.max_plies),
                repeat_limit=int(args.repeat_limit),
                repeat_min_ply=int(args.repeat_min_ply),
                no_capture_limit=int(args.no_capture_limit),
            )
            totals["samples"] += int(stats["samples"])
            totals["labeled_rows"] += int(stats["labeled_rows"])
            totals["entries"] += int(stats["entries"])
            totals["jobs"] += int(stats.get("jobs", 0))
            if stats["skipped"]:
                totals["skipped_shards"] += 1
                print(f"  {idx}/{len(shards)} {shard_path.name}: SKIP {stats.get('reason')}", flush=True)
            else:
                print(
                    f"  {idx}/{len(shards)} {shard_path.name}: "
                    f"rows={stats['labeled_rows']}/{stats['samples']} "
                    f"entries={stats['entries']} jobs={stats['jobs']} "
                    f"illegal={stats['illegal_candidates']} dt={stats['duration_s']:.0f}s",
                    flush=True,
                )
    finally:
        pool.close()

    print(f"DONE: {totals}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
