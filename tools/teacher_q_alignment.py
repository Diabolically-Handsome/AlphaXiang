"""Offline alignment diagnostic for teacher_q action-value shards.

For each position with teacher_q candidates, compare a checkpoint's policy to
the full-Pika candidate values:
  - model top-1 vs teacher_q top-1 agreement
  - teacher_q top-1 in model top-k
  - model policy rank of the teacher_q best move
  - regret of the model top-1 when that move is present in teacher_q candidates
  - listwise teacher_q cross entropy over the labeled candidate set

This is read-only and does not mutate shards.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


def _iter_shards(paths: list[Path], pattern: str) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file():
            out.append(path)
        else:
            out.extend(sorted(path.rglob(pattern)))
    return sorted(dict.fromkeys(path.resolve() for path in out if path.is_file()))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model


def _masked_log_probs(logits: torch.Tensor, legal_idxs: torch.Tensor) -> torch.Tensor:
    masked = torch.full_like(logits, -1.0e9)
    legal_idxs = legal_idxs.to(device=logits.device, dtype=torch.long)
    legal_idxs = legal_idxs[(legal_idxs >= 0) & (legal_idxs < logits.numel())]
    if legal_idxs.numel() == 0:
        return F.log_softmax(logits.float(), dim=0)
    masked[legal_idxs] = logits[legal_idxs]
    return F.log_softmax(masked.float(), dim=0)


def _analyze_shard(
    *,
    model,
    shard_path: Path,
    device: torch.device,
    batch_size: int,
    teacher_temperature_cp: float,
    top_k: int,
) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    states = payload["state"].to(torch.float32)
    n = int(states.shape[0])
    tq_offsets = payload.get("teacher_q_offsets")
    tq_idxs = payload.get("teacher_q_idxs")
    tq_values = payload.get("teacher_q_values")
    legal_offsets = payload.get("legal_offsets")
    legal_idxs = payload.get("legal_idxs")
    if tq_offsets is None or tq_idxs is None or tq_values is None:
        return {"path": str(shard_path), "samples": n, "teacher_q_rows": 0}
    if legal_offsets is None or legal_idxs is None:
        raise RuntimeError(f"missing legal CSR in {shard_path}")

    tq_offsets = tq_offsets.to(torch.long)
    tq_idxs = tq_idxs.to(torch.long)
    tq_values = tq_values.to(torch.float32)
    legal_offsets = legal_offsets.to(torch.long)
    legal_idxs = legal_idxs.to(torch.long)

    logits_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = states[start:stop].to(device, non_blocking=True)
            outputs = model(batch)
            logits_chunks.append(outputs["policy_logits"].detach().cpu().float())
    logits_all = torch.cat(logits_chunks, dim=0)

    rows = 0
    top1_hits = 0
    topk_hits = 0
    teacher_best_ranks: list[int] = []
    regrets: list[float] = []
    missing_model_top1 = 0
    ce_values: list[float] = []
    teacher_best_margins: list[float] = []
    examples: list[dict[str, Any]] = []

    fens = payload.get("fens")
    for i in range(n):
        start = int(tq_offsets[i].item())
        end = int(tq_offsets[i + 1].item())
        if end <= start:
            continue
        rows += 1
        cand_idxs = tq_idxs[start:end].to(torch.long)
        cand_values = tq_values[start:end].to(torch.float32)
        best_pos = int(torch.argmax(cand_values).item())
        teacher_best = int(cand_idxs[best_pos].item())
        best_value = float(cand_values[best_pos].item())
        if cand_values.numel() >= 2:
            top2 = torch.topk(cand_values, k=2).values
            teacher_best_margins.append(float((top2[0] - top2[1]).item()))

        leg_start = int(legal_offsets[i].item())
        leg_end = int(legal_offsets[i + 1].item())
        legal = legal_idxs[leg_start:leg_end]
        log_probs = _masked_log_probs(logits_all[i], legal)
        probs = log_probs.exp()
        legal_probs = probs[legal]
        order = torch.argsort(legal_probs, descending=True)
        legal_ordered = legal[order]
        model_top = int(legal_ordered[0].item()) if legal_ordered.numel() else int(torch.argmax(probs).item())
        topk = {int(x) for x in legal_ordered[:top_k].tolist()}
        if model_top == teacher_best:
            top1_hits += 1
        if teacher_best in topk:
            topk_hits += 1
        rank_matches = (legal_ordered == teacher_best).nonzero(as_tuple=False)
        if int(rank_matches.numel()) > 0:
            teacher_best_ranks.append(int(rank_matches[0].item()) + 1)

        cand_map = {int(idx): float(value) for idx, value in zip(cand_idxs.tolist(), cand_values.tolist())}
        if model_top in cand_map:
            regrets.append(best_value - cand_map[model_top])
        else:
            missing_model_top1 += 1

        teacher_probs = torch.softmax(cand_values / float(teacher_temperature_cp), dim=0)
        candidate_log_probs = log_probs[cand_idxs.clamp(0, log_probs.numel() - 1)]
        ce = float((-(teacher_probs.to(candidate_log_probs.device) * candidate_log_probs.cpu()).sum()).item())
        ce_values.append(ce)

        if len(examples) < 8 and model_top != teacher_best:
            examples.append({
                "row": i,
                "fen": None if not isinstance(fens, list) else fens[i],
                "model_top1": model_top,
                "teacher_top1": teacher_best,
                "teacher_best_value_cp": best_value,
                "teacher_best_rank": teacher_best_ranks[-1] if teacher_best_ranks else None,
                "model_top1_regret_cp": regrets[-1] if model_top in cand_map else None,
            })

    return {
        "path": str(shard_path),
        "samples": n,
        "teacher_q_rows": rows,
        "top1_agreement": top1_hits / rows if rows else None,
        f"teacher_top1_in_model_top{top_k}": topk_hits / rows if rows else None,
        "median_teacher_best_rank": median(teacher_best_ranks) if teacher_best_ranks else None,
        "p75_teacher_best_rank": _percentile([float(x) for x in teacher_best_ranks], 0.75),
        "model_top1_in_teacher_candidates": (rows - missing_model_top1) / rows if rows else None,
        "median_model_top1_regret_cp": median(regrets) if regrets else None,
        "p75_model_top1_regret_cp": _percentile(regrets, 0.75),
        "median_teacher_q_ce": median(ce_values) if ce_values else None,
        "median_teacher_best_margin_cp": median(teacher_best_margins) if teacher_best_margins else None,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint policy alignment on teacher_q shards.")
    parser.add_argument("paths", nargs="+", help="Shard files or directories")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pattern", default="shard_*.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-temperature-cp", type=float, default=80.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    shard_paths = _iter_shards([Path(p) for p in args.paths], args.pattern)
    if not shard_paths:
        raise SystemExit("no shards found")

    device = torch.device(args.device)
    model = _load_model(Path(args.checkpoint), device)
    rows = [
        _analyze_shard(
            model=model,
            shard_path=path,
            device=device,
            batch_size=int(args.batch_size),
            teacher_temperature_cp=float(args.teacher_temperature_cp),
            top_k=int(args.top_k),
        )
        for path in shard_paths
    ]

    total_rows = sum(int(row.get("teacher_q_rows", 0)) for row in rows)
    weighted: dict[str, float | None] = {}
    for key in [
        "top1_agreement",
        f"teacher_top1_in_model_top{int(args.top_k)}",
        "model_top1_in_teacher_candidates",
    ]:
        numerator = 0.0
        denom = 0
        for row in rows:
            value = row.get(key)
            count = int(row.get("teacher_q_rows", 0))
            if value is None or count <= 0:
                continue
            numerator += float(value) * count
            denom += count
        weighted[key] = numerator / denom if denom else None

    all_regrets = [float(x) for row in rows for x in ([] if row.get("median_model_top1_regret_cp") is None else [])]
    # Medians need the original per-row lists; keep aggregate medians as median of shard medians
    # to avoid large JSON output. Shards are similarly sized in current pipelines.
    shard_median_regrets = [
        float(row["median_model_top1_regret_cp"])
        for row in rows
        if row.get("median_model_top1_regret_cp") is not None
    ]
    shard_median_ce = [
        float(row["median_teacher_q_ce"])
        for row in rows
        if row.get("median_teacher_q_ce") is not None
    ]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "shards": len(rows),
        "teacher_q_rows": total_rows,
        **weighted,
        "median_of_shard_median_model_top1_regret_cp": (
            median(shard_median_regrets) if shard_median_regrets else None
        ),
        "median_of_shard_median_teacher_q_ce": median(shard_median_ce) if shard_median_ce else None,
    }
    result = {"summary": summary, "shards_detail": rows}

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
