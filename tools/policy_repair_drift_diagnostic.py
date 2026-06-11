"""Compare a repair checkpoint against a frozen reference on shard positions.

This is the offline gate for V13-style local repair.  It answers the questions
we care about before spending GPU hours on arena games:

* Did the repair move teacher-Q good moves above bad moves?
* Did ordinary anchor positions drift away from the reference policy?
* Did value_scalar move even when we only intended a policy repair?

The script is read-only: it never edits checkpoints or shards.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from hard_position_mining import _load_model, predict_values_and_logits  # noqa: E402
from xiangqi_train import _policy_log_probs_with_optional_legal_mask  # noqa: E402


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    ordered = sorted(xs)
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    t = pos - lo
    return float(ordered[lo] * (1.0 - t) + ordered[hi] * t)


def _select_indices(n: int, max_samples: int | None, stride: int) -> torch.Tensor:
    stride = max(1, int(stride))
    idxs = torch.arange(0, n, stride, dtype=torch.long)
    if max_samples is not None and int(max_samples) > 0 and idxs.numel() > int(max_samples):
        # Deterministic coverage across the shard instead of just taking the head.
        keep = torch.linspace(0, idxs.numel() - 1, steps=int(max_samples)).round().long()
        idxs = idxs[keep]
    return idxs.contiguous()


def _slice_csr(
    offsets: torch.Tensor | None,
    values: torch.Tensor | None,
    rows: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if offsets is None or values is None:
        return None, None
    offsets = offsets.to(torch.int64).cpu()
    chunks: list[torch.Tensor] = []
    out_offsets = [0]
    for row in rows.tolist():
        start = int(offsets[int(row)].item())
        end = int(offsets[int(row) + 1].item())
        chunk = values[start:end].cpu()
        chunks.append(chunk)
        out_offsets.append(out_offsets[-1] + int(chunk.numel()))
    if chunks:
        flat = torch.cat(chunks, dim=0).contiguous()
    else:
        flat = torch.empty(0, dtype=values.dtype)
    return torch.tensor(out_offsets, dtype=torch.int64), flat


def _slice_optional_vector(values: torch.Tensor | None, rows: torch.Tensor) -> torch.Tensor | None:
    if values is None:
        return None
    return values[rows].cpu().contiguous()


def _masked_argmax(log_probs: torch.Tensor) -> torch.Tensor:
    return torch.argmax(log_probs, dim=1)


def _topk_overlap(log_probs_a: torch.Tensor, log_probs_b: torch.Tensor, k: int) -> list[float]:
    k = max(1, int(k))
    out: list[float] = []
    top_a = torch.topk(log_probs_a, k=min(k, log_probs_a.shape[1]), dim=1).indices
    top_b = torch.topk(log_probs_b, k=min(k, log_probs_b.shape[1]), dim=1).indices
    for a, b in zip(top_a, top_b):
        sa = {int(x) for x in a.tolist() if int(x) >= 0}
        sb = {int(x) for x in b.tolist() if int(x) >= 0}
        denom = max(1, min(len(sa), len(sb), k))
        out.append(len(sa & sb) / float(denom))
    return out


def _policy_kl(ref_log_probs: torch.Tensor, cand_log_probs: torch.Tensor) -> torch.Tensor:
    ref_probs = ref_log_probs.exp()
    finite = torch.isfinite(ref_log_probs) & torch.isfinite(cand_log_probs)
    terms = torch.where(finite, ref_probs * (ref_log_probs - cand_log_probs), torch.zeros_like(ref_probs))
    return terms.sum(dim=1)


def _teacher_q_metrics(
    *,
    ref_log_probs: torch.Tensor,
    cand_log_probs: torch.Tensor,
    teacher_q_offsets: torch.Tensor | None,
    teacher_q_idxs: torch.Tensor | None,
    teacher_q_values: torch.Tensor | None,
    bad_move: torch.Tensor | None,
    min_gap_cp: float,
) -> dict[str, Any]:
    if teacher_q_offsets is None or teacher_q_idxs is None or teacher_q_values is None:
        return {"teacher_q_rows": 0}

    rows = 0
    gap_improvements: list[float] = []
    best_prob_changes: list[float] = []
    bad_prob_changes: list[float] = []
    ref_regrets: list[float] = []
    cand_regrets: list[float] = []
    best_rank_ref: list[float] = []
    best_rank_cand: list[float] = []
    known_bad_gap_improvements: list[float] = []
    known_bad_prob_changes: list[float] = []
    known_bad_logp_changes: list[float] = []
    known_bad_ref_gaps: list[float] = []
    known_bad_cand_gaps: list[float] = []
    known_bad_rank_ref: list[float] = []
    known_bad_rank_cand: list[float] = []
    min_gap = max(0.0, float(min_gap_cp))

    for i in range(int(ref_log_probs.shape[0])):
        start = int(teacher_q_offsets[i].item())
        end = int(teacher_q_offsets[i + 1].item())
        if end - start < 2:
            continue
        idxs = teacher_q_idxs[start:end].to(torch.long)
        values = teacher_q_values[start:end].float()
        if idxs.numel() < 2:
            continue
        if int(idxs.min().item()) < 0 or int(idxs.max().item()) >= ref_log_probs.shape[1]:
            continue

        best_pos = int(torch.argmax(values).item())
        best_value = float(values[best_pos].item())
        gaps = best_value - values
        bad_mask = gaps >= min_gap
        bad_mask[best_pos] = False
        if not bool(bad_mask.any()):
            continue

        ref_selected = ref_log_probs[i, idxs].float()
        cand_selected = cand_log_probs[i, idxs].float()
        if bool((ref_selected < -1e8).any()) or bool((cand_selected < -1e8).any()):
            continue

        rows += 1
        bad_positions = torch.where(bad_mask)[0]
        # Track the worst teacher-Q candidate that the reference liked most.
        ref_bad_pos = int(bad_positions[torch.argmax(ref_selected[bad_positions])].item())
        ref_gap = float((ref_selected[best_pos] - ref_selected[ref_bad_pos]).item())
        cand_gap = float((cand_selected[best_pos] - cand_selected[ref_bad_pos]).item())
        gap_improvements.append(cand_gap - ref_gap)
        best_prob_changes.append(float((cand_selected[best_pos].exp() - ref_selected[best_pos].exp()).item()))
        bad_prob_changes.append(float((cand_selected[ref_bad_pos].exp() - ref_selected[ref_bad_pos].exp()).item()))

        ref_choice = int(torch.argmax(ref_selected).item())
        cand_choice = int(torch.argmax(cand_selected).item())
        ref_regrets.append(float(max(0.0, best_value - float(values[ref_choice].item()))))
        cand_regrets.append(float(max(0.0, best_value - float(values[cand_choice].item()))))

        ref_order = torch.argsort(ref_selected, descending=True)
        cand_order = torch.argsort(cand_selected, descending=True)
        best_rank_ref.append(float((ref_order == best_pos).nonzero(as_tuple=False)[0].item() + 1))
        best_rank_cand.append(float((cand_order == best_pos).nonzero(as_tuple=False)[0].item() + 1))

        if bad_move is not None:
            known_bad = int(bad_move[i].item())
            if known_bad >= 0:
                matches = (idxs == known_bad).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    known_bad_pos = int(matches[0].item())
                    if known_bad_pos != best_pos and float(gaps[known_bad_pos].item()) >= min_gap:
                        ref_known_gap = float((ref_selected[best_pos] - ref_selected[known_bad_pos]).item())
                        cand_known_gap = float((cand_selected[best_pos] - cand_selected[known_bad_pos]).item())
                        known_bad_ref_gaps.append(ref_known_gap)
                        known_bad_cand_gaps.append(cand_known_gap)
                        known_bad_gap_improvements.append(cand_known_gap - ref_known_gap)
                        known_bad_prob_changes.append(
                            float((cand_selected[known_bad_pos].exp() - ref_selected[known_bad_pos].exp()).item())
                        )
                        known_bad_logp_changes.append(
                            float((cand_selected[known_bad_pos] - ref_selected[known_bad_pos]).item())
                        )
                        known_bad_rank_ref.append(
                            float((ref_order == known_bad_pos).nonzero(as_tuple=False)[0].item() + 1)
                        )
                        known_bad_rank_cand.append(
                            float((cand_order == known_bad_pos).nonzero(as_tuple=False)[0].item() + 1)
                        )

    return {
        "teacher_q_rows": rows,
        "median_good_vs_bad_logp_gap_improvement": _median(gap_improvements),
        "mean_good_vs_bad_logp_gap_improvement": _mean(gap_improvements),
        "median_best_prob_change": _median(best_prob_changes),
        "median_ref_liked_bad_prob_change": _median(bad_prob_changes),
        "median_ref_regret_cp": _median(ref_regrets),
        "median_candidate_regret_cp": _median(cand_regrets),
        "median_regret_delta_cp": (
            _median(cand_regrets) - _median(ref_regrets)
            if _median(cand_regrets) is not None and _median(ref_regrets) is not None
            else None
        ),
        "median_teacher_best_rank_reference": _median(best_rank_ref),
        "median_teacher_best_rank_candidate": _median(best_rank_cand),
        "known_bad_rows": len(known_bad_gap_improvements),
        "median_known_bad_gap_improvement": _median(known_bad_gap_improvements),
        "mean_known_bad_gap_improvement": _mean(known_bad_gap_improvements),
        "median_known_bad_prob_change": _median(known_bad_prob_changes),
        "median_known_bad_logp_change": _median(known_bad_logp_changes),
        "median_teacher_best_vs_known_bad_gap_reference": _median(known_bad_ref_gaps),
        "median_teacher_best_vs_known_bad_gap_candidate": _median(known_bad_cand_gaps),
        "median_known_bad_rank_reference": _median(known_bad_rank_ref),
        "median_known_bad_rank_candidate": _median(known_bad_rank_cand),
    }


def evaluate_shard(
    *,
    shard_path: Path,
    ref_model,
    cand_model,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    sample_stride: int,
    use_bfloat16: bool,
    min_gap_cp: float,
) -> dict[str, Any]:
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    state = payload.get("state")
    if not isinstance(state, torch.Tensor):
        return {"path": str(shard_path), "samples": 0, "skipped": True, "reason": "missing state tensor"}
    n = int(state.shape[0])
    rows = _select_indices(n, max_samples=max_samples, stride=sample_stride)
    if rows.numel() == 0:
        return {"path": str(shard_path), "samples": 0, "skipped": True, "reason": "empty selection"}

    state_sel = state[rows].to(torch.float32).contiguous()
    legal_offsets, legal_idxs = _slice_csr(payload.get("legal_offsets"), payload.get("legal_idxs"), rows)
    teacher_q_offsets, teacher_q_idxs = _slice_csr(
        payload.get("teacher_q_offsets"), payload.get("teacher_q_idxs"), rows
    )
    _teacher_q_offsets2, teacher_q_values = _slice_csr(
        payload.get("teacher_q_offsets"), payload.get("teacher_q_values"), rows
    )
    bad_move = _slice_optional_vector(payload.get("bad_move"), rows)

    ref_values, ref_logits = predict_values_and_logits(
        ref_model, state_sel, device, batch_size=batch_size, use_bfloat16=use_bfloat16
    )
    cand_values, cand_logits = predict_values_and_logits(
        cand_model, state_sel, device, batch_size=batch_size, use_bfloat16=use_bfloat16
    )

    ref_log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=ref_logits.to(device),
        legal_offsets=legal_offsets.to(device) if legal_offsets is not None else None,
        legal_idxs=legal_idxs.to(device) if legal_idxs is not None else None,
    ).cpu()
    cand_log_probs = _policy_log_probs_with_optional_legal_mask(
        policy_logits=cand_logits.to(device),
        legal_offsets=legal_offsets.to(device) if legal_offsets is not None else None,
        legal_idxs=legal_idxs.to(device) if legal_idxs is not None else None,
    ).cpu()

    kl = _policy_kl(ref_log_probs, cand_log_probs).tolist()
    ref_top1 = _masked_argmax(ref_log_probs)
    cand_top1 = _masked_argmax(cand_log_probs)
    top1_change_rate = float((ref_top1 != cand_top1).float().mean().item())
    value_drift = (cand_values - ref_values).abs().tolist()

    metrics = {
        "path": str(shard_path),
        "samples": int(rows.numel()),
        "source_samples": n,
        "skipped": False,
        "policy_kl_mean": _mean([float(x) for x in kl]),
        "policy_kl_median": _median([float(x) for x in kl]),
        "policy_kl_p95": _percentile([float(x) for x in kl], 0.95),
        "top1_change_rate": top1_change_rate,
        "top3_overlap_mean": _mean(_topk_overlap(ref_log_probs, cand_log_probs, 3)),
        "top5_overlap_mean": _mean(_topk_overlap(ref_log_probs, cand_log_probs, 5)),
        "value_abs_drift_mean": _mean([float(x) for x in value_drift]),
        "value_abs_drift_median": _median([float(x) for x in value_drift]),
        "value_abs_drift_p95": _percentile([float(x) for x in value_drift], 0.95),
    }
    metrics.update(
        _teacher_q_metrics(
            ref_log_probs=ref_log_probs,
            cand_log_probs=cand_log_probs,
            teacher_q_offsets=teacher_q_offsets,
            teacher_q_idxs=teacher_q_idxs,
            teacher_q_values=teacher_q_values,
            bad_move=bad_move,
            min_gap_cp=min_gap_cp,
        )
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline drift gate for V13 policy repair checkpoints.")
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--shard-glob", default="shard_*.pt")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-samples-per-shard", type=int, default=512)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--teacher-q-min-gap-cp", type=float, default=80.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob(args.shard_glob))
    if int(args.max_shards) > 0:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise SystemExit(f"no shards matching {args.shard_glob!r} under {shard_dir}")

    device = torch.device(args.device)
    ref_model = _load_model(Path(args.reference_checkpoint), device)
    cand_model = _load_model(Path(args.candidate_checkpoint), device)

    rows = []
    for shard in shards:
        print(f"diagnosing {shard}", flush=True)
        rows.append(
            evaluate_shard(
                shard_path=shard,
                ref_model=ref_model,
                cand_model=cand_model,
                device=device,
                batch_size=int(args.batch_size),
                max_samples=int(args.max_samples_per_shard) if int(args.max_samples_per_shard) > 0 else None,
                sample_stride=int(args.sample_stride),
                use_bfloat16=not bool(args.disable_bf16),
                min_gap_cp=float(args.teacher_q_min_gap_cp),
            )
        )

    valid = [row for row in rows if not row.get("skipped")]
    summary = {
        "reference_checkpoint": str(args.reference_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "shard_dir": str(shard_dir),
        "shards": len(rows),
        "valid_shards": len(valid),
        "samples": sum(int(row.get("samples", 0)) for row in valid),
        "policy_kl_mean_of_shards": _mean([float(row["policy_kl_mean"]) for row in valid if row.get("policy_kl_mean") is not None]),
        "policy_kl_p95_of_shards": _mean([float(row["policy_kl_p95"]) for row in valid if row.get("policy_kl_p95") is not None]),
        "top1_change_rate_mean_of_shards": _mean([float(row["top1_change_rate"]) for row in valid if row.get("top1_change_rate") is not None]),
        "top5_overlap_mean_of_shards": _mean([float(row["top5_overlap_mean"]) for row in valid if row.get("top5_overlap_mean") is not None]),
        "value_abs_drift_mean_of_shards": _mean([float(row["value_abs_drift_mean"]) for row in valid if row.get("value_abs_drift_mean") is not None]),
        "teacher_q_rows": sum(int(row.get("teacher_q_rows", 0)) for row in valid),
        "teacher_q_median_regret_delta_cp_of_shards": _mean([
            float(row["median_regret_delta_cp"])
            for row in valid
            if row.get("median_regret_delta_cp") is not None
        ]),
        "per_shard": rows,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
