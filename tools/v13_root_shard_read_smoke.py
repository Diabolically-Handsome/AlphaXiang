#!/usr/bin/env python3
"""Lightweight smoke for V13 root-regret tensor shards.

This avoids a full xiangqi_train.py run, which always performs a final human-val
pass at max_steps.  It still exercises the trainer's tensorized shard reader,
collator, legal masks, teacher-Q pairwise loss, and bad-move suppression loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_train import (  # noqa: E402
    _collate_sample_blobs,
    _compute_training_losses,
    _extract_tensorized_sample_blobs,
    _get_selfplay_shard_sample_count,
    _move_batch_to_device,
)
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


def _load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_state(state)
    model.to(device)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--disable-bf16", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    shard_path = Path(args.shard)
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    sample_count = _get_selfplay_shard_sample_count(shard, shard_path)
    take = min(int(args.batch_size), int(sample_count))
    blobs = _extract_tensorized_sample_blobs(shard, list(range(take)))
    batch = _move_batch_to_device(_collate_sample_blobs(blobs), device)

    model = _load_model(Path(args.checkpoint), device)
    anchor_model = _load_model(Path(args.anchor_checkpoint or args.checkpoint), device)
    autocast_enabled = bool(not args.disable_bf16 and device.type == "cuda")
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = model(batch["state"])
            anchor_outputs = anchor_model(batch["state"])
            losses = _compute_training_losses(
                policy_logits=outputs["policy_logits"],
                value_scalar=outputs["value_scalar"],
                policy_offsets=batch["policy_offsets"],
                policy_idxs=batch["policy_idxs"],
                policy_probs=batch["policy_probs"],
                z=batch["z"],
                wdl_logits=outputs.get("wdl_logits"),
                wdl_target=batch.get("wdl_target"),
                wdl_loss_weight=0.0,
                policy_loss_weight=0.0,
                value_loss_weight=0.0,
                wdl_value_consistency_weight=0.0,
                oracle_value=batch.get("oracle_value"),
                use_oracle_value=True,
                teacher_q_offsets=batch.get("teacher_q_offsets"),
                teacher_q_idxs=batch.get("teacher_q_idxs"),
                teacher_q_values=batch.get("teacher_q_values"),
                teacher_q_loss_weight=0.0,
                teacher_q_pairwise_loss_weight=1.0,
                teacher_q_pairwise_margin_logit=0.35,
                teacher_q_pairwise_min_gap_cp=150.0,
                teacher_q_pairwise_beta=1.0,
                teacher_q_ref_policy_logits=anchor_outputs["policy_logits"],
                teacher_q_pairwise_bad_move_only=True,
                bad_move_suppression_loss_weight=0.5,
                bad_move_suppression_margin_logit=0.75,
                bad_move_suppression_min_gap_cp=150.0,
                bad_move_suppression_beta=2.0,
                bad_move=batch.get("bad_move"),
                sample_weight=batch.get("sample_weight"),
                legal_offsets=batch.get("legal_offsets"),
                legal_idxs=batch.get("legal_idxs"),
            )

    summary = {
        "shard": str(shard_path),
        "samples_in_shard": int(sample_count),
        "samples_checked": int(take),
        "has_teacher_q": bool(batch.get("teacher_q_idxs") is not None and batch["teacher_q_idxs"].numel() > 0),
        "has_legal_mask": bool(batch.get("legal_idxs") is not None and batch["legal_idxs"].numel() > 0),
        "has_bad_move": bool(batch.get("bad_move") is not None and (batch["bad_move"] >= 0).any().item()),
        "has_oracle_value": bool(batch.get("oracle_value") is not None),
        "oracle_value_coverage": (
            0.0
            if batch.get("oracle_value") is None
            else float(torch.isfinite(batch["oracle_value"]).float().mean().detach().cpu().item())
        ),
        "teacher_q_pairwise_loss": float(losses["teacher_q_pairwise_loss"].detach().cpu().item()),
        "bad_move_suppression_loss": float(losses["bad_move_suppression_loss"].detach().cpu().item()),
        "n_teacher_q_pairwise_samples": int(losses["n_teacher_q_pairwise_samples"].detach().cpu().item()),
        "n_bad_move_suppression_samples": int(losses["n_bad_move_suppression_samples"].detach().cpu().item()),
        "n_oracle_samples": int(losses["n_oracle_samples"].detach().cpu().item()),
        "total_loss": float(losses["total_loss"].detach().cpu().item()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["has_teacher_q"] or not summary["has_legal_mask"]:
        raise SystemExit("shard is missing teacher_q or legal mask fields")
    if not summary["has_oracle_value"] or summary["oracle_value_coverage"] < 1.0:
        raise SystemExit("shard is missing full oracle_value coverage")
    if summary["has_bad_move"] and summary["n_bad_move_suppression_samples"] <= 0:
        raise SystemExit("bad_move labels exist, but suppression loss found no valid sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
