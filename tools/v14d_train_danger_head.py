#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xiangqi_danger_model import CNNActionDangerHead, DangerHeadConfig, save_danger_checkpoint  # noqa: E402


def _split_by_group(group_ids: Tensor, val_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    groups = sorted({int(v) for v in group_ids.tolist()})
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * float(val_fraction)))) if len(groups) > 1 else 0
    val_groups = set(groups[:n_val])
    val_mask = torch.tensor([int(g) in val_groups for g in group_ids.tolist()], dtype=torch.bool)
    train_mask = ~val_mask
    if int(train_mask.sum().item()) <= 0:
        train_mask[:] = True
        val_mask[:] = False
    return train_mask, val_mask


def _auc_score(scores: Tensor, labels: Tensor) -> float | None:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    wins = (pos[:, None] > neg[None, :]).float().sum()
    ties = (pos[:, None] == neg[None, :]).float().sum()
    return float((wins + 0.5 * ties) / float(pos.numel() * neg.numel()))


def _pairwise_loss(scores: Tensor, group_ids: Tensor, labels: Tensor, margin: float) -> Tensor:
    losses: list[Tensor] = []
    for group in torch.unique(group_ids):
        mask = group_ids == group
        pos = scores[mask & (labels > 0.5)]
        neg = scores[mask & (labels <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.softplus(-(pos[:, None] - neg[None, :] - float(margin))).mean())
    if not losses:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: CNNActionDangerHead, data: dict[str, Tensor], indices: Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    if indices.numel() == 0:
        return {}
    states = data["state_after"][indices].to(device=device, dtype=torch.float32)
    labels = data["labels"][indices].to(device=device, dtype=torch.float32)
    group_ids = data["group_id"][indices].to(device=device, dtype=torch.long)
    logits = model(states)["danger_logits"]
    risk = model(states)["risk_logit"]
    bce = F.binary_cross_entropy_with_logits(logits, labels).item()
    refuted = labels[:, 4]
    risk_prob = torch.sigmoid(risk)
    pred = (risk_prob >= 0.5).float()
    tp = float(((pred > 0.5) & (refuted > 0.5)).sum().item())
    fp = float(((pred > 0.5) & (refuted <= 0.5)).sum().item())
    fn = float(((pred <= 0.5) & (refuted > 0.5)).sum().item())
    tn = float(((pred <= 0.5) & (refuted <= 0.5)).sum().item())
    recall = tp / max(1.0, tp + fn)
    precision = tp / max(1.0, tp + fp)
    fpr = fp / max(1.0, fp + tn)
    auc = _auc_score(risk, refuted)

    pair_total = 0
    pair_correct = 0
    top_positive = 0
    top_total = 0
    for group in torch.unique(group_ids):
        mask = group_ids == group
        group_risk = risk[mask]
        group_label = refuted[mask]
        pos = group_risk[group_label > 0.5]
        neg = group_risk[group_label <= 0.5]
        if pos.numel() > 0 and neg.numel() > 0:
            pair_total += int(pos.numel() * neg.numel())
            pair_correct += int((pos[:, None] > neg[None, :]).sum().item())
            top_total += 1
            top_idx = int(torch.argmax(group_risk).item())
            if float(group_label[top_idx].item()) > 0.5:
                top_positive += 1

    return {
        "bce": float(bce),
        "auc": auc,
        "recall_at_0p5": float(recall),
        "precision_at_0p5": float(precision),
        "false_positive_rate_at_0p5": float(fpr),
        "pair_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
        "top_risk_positive_rate": None if top_total == 0 else float(top_positive / top_total),
        "samples": int(indices.numel()),
        "positive_samples": int(refuted.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V14D CNN action danger head.")
    parser.add_argument("--dataset", default="/home/laure/alphaxiang/v14d_danger_data/danger_dataset.pt")
    parser.add_argument("--output-dir", default="/home/laure/alphaxiang/training_runs/run_042a_v14d_action_danger_head")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    data = {
        "state_after": payload["state_after"].to(torch.float32),
        "labels": payload["labels"].to(torch.float32),
        "group_id": payload["group_id"].to(torch.long),
    }
    train_mask, val_mask = _split_by_group(data["group_id"], float(args.val_fraction), int(args.seed))
    train_idx = train_mask.nonzero(as_tuple=False).flatten()
    val_idx = val_mask.nonzero(as_tuple=False).flatten()

    config = DangerHeadConfig(channels=int(args.channels), blocks=int(args.blocks), dropout=float(args.dropout))
    model = CNNActionDangerHead(config).to(device)

    train_ds = TensorDataset(
        data["state_after"][train_idx],
        data["labels"][train_idx],
        data["group_id"][train_idx],
    )
    loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    pos_counts = data["labels"][train_idx].sum(dim=0)
    neg_counts = float(train_idx.numel()) - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp_min(1.0)).clamp(1.0, 20.0).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    best_score = -math.inf
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for states, labels, group_ids in loader:
            states = states.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            group_ids = group_ids.to(device=device, dtype=torch.long)
            out = model(states)
            logits = out["danger_logits"]
            risk = out["risk_logit"]
            bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            pair = _pairwise_loss(risk, group_ids, labels[:, 4], margin=float(args.pairwise_margin))
            loss = bce + float(args.pairwise_weight) * pair
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())
            batches += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            train_metrics = evaluate(model, data, train_idx, device)
            val_metrics = evaluate(model, data, val_idx, device) if val_idx.numel() > 0 else {}
            selector = val_metrics or train_metrics
            score = float(selector.get("pair_accuracy") or 0.0) + float(selector.get("top_risk_positive_rate") or 0.0)
            entry = {
                "epoch": int(epoch),
                "loss": total_loss / max(1, batches),
                "train": train_metrics,
                "val": val_metrics,
                "score": score,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            if score > best_score:
                best_score = score
                best_metrics = entry
                save_danger_checkpoint(
                    str(output_dir / "best.pt"),
                    model=model,
                    extra={
                        "epoch": int(epoch),
                        "metrics": entry,
                        "dataset": str(args.dataset),
                    },
                )

    save_danger_checkpoint(
        str(output_dir / "latest.pt"),
        model=model,
        extra={
            "epoch": int(args.epochs),
            "best_metrics": best_metrics,
            "dataset": str(args.dataset),
        },
    )
    summary = {
        "output_dir": str(output_dir),
        "best_score": best_score,
        "best_metrics": best_metrics,
        "train_samples": int(train_idx.numel()),
        "val_samples": int(val_idx.numel()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
