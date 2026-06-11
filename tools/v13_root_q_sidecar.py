#!/usr/bin/env python3
"""Train a lightweight tabular sidecar for V13 root-Q error risk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SidecarMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.refute_head = nn.Linear(hidden_dim // 2, 1)
        self.regret_head = nn.Linear(hidden_dim // 2, 1)
        self.stop_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.refute_head(h).squeeze(-1), self.regret_head(h).squeeze(-1), self.stop_head(h).squeeze(-1)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["root_key"])].append(row)
    return grouped


def _root_is_bad(rows: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("candidate_is_selected")) and bool(row.get("root_selected_bad")) for row in rows)


def _split_keys(grouped: dict[str, list[dict[str, Any]]], train_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    bad = [key for key, rows in grouped.items() if _root_is_bad(rows)]
    clean = [key for key, rows in grouped.items() if not _root_is_bad(rows)]
    rng = random.Random(seed)
    rng.shuffle(bad)
    rng.shuffle(clean)

    def split(items: list[str]) -> tuple[list[str], list[str]]:
        if len(items) <= 1:
            return items, []
        n_train = max(1, min(len(items) - 1, int(round(len(items) * train_fraction))))
        return items[:n_train], items[n_train:]

    bad_train, bad_val = split(bad)
    clean_train, clean_val = split(clean)
    return set(bad_train + clean_train), set(bad_val + clean_val)


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update((row.get("features") or {}).keys())
    return sorted(names)


def _rows_for_keys(grouped: dict[str, list[dict[str, Any]]], keys: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(keys):
        out.extend(grouped[key])
    return out


def _tensorize(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    mean_t: torch.Tensor | None = None,
    std_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    y_ref: list[float] = []
    y_reg: list[float] = []
    weights: list[float] = []
    selected: list[float] = []
    root_bad: list[float] = []
    max_log = math.log1p(20000.0)
    for row in rows:
        feats = row.get("features") or {}
        xs.append([float(feats.get(name, 0.0)) for name in feature_names])
        regret = max(min(float(row.get("candidate_regret_cp", 0.0) or 0.0), 20000.0), 0.0)
        refuted = float(regret >= 150.0)
        is_selected = float(bool(row.get("candidate_is_selected")))
        is_root_bad = float(bool(row.get("root_selected_bad")))
        is_cat = float(bool(row.get("root_catastrophic")))
        y_ref.append(refuted)
        y_reg.append(math.log1p(regret) / max_log)
        weights.append(1.0 + 3.0 * refuted + 5.0 * is_selected * is_root_bad + 3.0 * is_cat)
        selected.append(is_selected)
        root_bad.append(is_root_bad)
    x = torch.tensor(xs, dtype=torch.float32)
    if mean_t is None:
        mean_t = x.mean(dim=0)
    if std_t is None:
        std_t = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean_t) / std_t
    return (
        x,
        torch.tensor(y_ref, dtype=torch.float32),
        torch.tensor(y_reg, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(selected, dtype=torch.float32),
        torch.tensor(root_bad, dtype=torch.float32),
    )


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [(s, l) for s, l in zip(scores, labels) if l == 1]
    neg = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    neg_scores = [s for s, _ in neg]
    for s, _ in pos:
        for n in neg_scores:
            total += 1.0
            if s > n:
                wins += 1.0
            elif s == n:
                wins += 0.5
    return wins / total if total else None


@torch.no_grad()
def _predict(model: SidecarMLP, x: torch.Tensor, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs: list[torch.Tensor] = []
    regrets: list[torch.Tensor] = []
    stops: list[torch.Tensor] = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device)
        ref_logit, reg_logit, stop_logit = model(xb)
        probs.append(torch.sigmoid(ref_logit).cpu())
        regrets.append(torch.sigmoid(reg_logit).cpu())
        stops.append(torch.sigmoid(stop_logit).cpu())
    return torch.cat(probs), torch.cat(regrets), torch.cat(stops)


def _metrics(rows: list[dict[str, Any]], probs: torch.Tensor, reg_pred: torch.Tensor, stop_probs: torch.Tensor) -> dict[str, Any]:
    labels = [int(float(row.get("candidate_regret_cp", 0.0) or 0.0) >= 150.0) for row in rows]
    scores = [float(x) for x in probs.tolist()]
    stop_scores = [float(x) for x in stop_probs.tolist()]
    reg_targets = [math.log1p(max(min(float(row.get("candidate_regret_cp", 0.0) or 0.0), 20000.0), 0.0)) / math.log1p(20000.0) for row in rows]
    mae = [abs(float(p) - float(t)) for p, t in zip(reg_pred.tolist(), reg_targets)]
    selected_scores = [
        float(score)
        for row, score in zip(rows, stop_scores)
        if bool(row.get("candidate_is_selected"))
    ]
    bad_selected_scores = [
        float(score)
        for row, score in zip(rows, stop_scores)
        if bool(row.get("candidate_is_selected")) and bool(row.get("root_selected_bad"))
    ]
    clean_selected_scores = [
        float(score)
        for row, score in zip(rows, stop_scores)
        if bool(row.get("candidate_is_selected")) and not bool(row.get("root_selected_bad"))
    ]
    selected_labels = [
        int(bool(row.get("root_selected_bad")))
        for row in rows
        if bool(row.get("candidate_is_selected"))
    ]
    return {
        "rows": len(rows),
        "roots": len({row["root_key"] for row in rows}),
        "refuted_rows": sum(labels),
        "candidate_auc": _auc(scores, labels),
        "selected_stop_auc": _auc(selected_scores, selected_labels),
        "regret_log_mae": None if not mae else mean(mae),
        "selected_risk_median": None if not selected_scores else median(selected_scores),
        "bad_selected_risk_median": None if not bad_selected_scores else median(bad_selected_scores),
        "clean_selected_risk_median": None if not clean_selected_scores else median(clean_selected_scores),
    }


def _write_predictions(path: Path, rows: list[dict[str, Any]], probs: torch.Tensor, reg_pred: torch.Tensor, stop_probs: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, p, r, s in zip(rows, probs.tolist(), reg_pred.tolist(), stop_probs.tolist()):
            out = {
                "root_key": row["root_key"],
                "candidate_move": row["candidate_move"],
                "selected_move": row["selected_move"],
                "teacher_best_move": row["teacher_best_move"],
                "candidate_is_selected": row["candidate_is_selected"],
                "root_selected_bad": row["root_selected_bad"],
                "root_catastrophic": row["root_catastrophic"],
                "root_class": row.get("root_class"),
                "candidate_regret_cp": row["candidate_regret_cp"],
                "candidate_child_d16_score_cp": row["candidate_child_d16_score_cp"],
                "pred_refute_prob": float(p),
                "pred_regret_log01": float(r),
                "pred_stop_prob": float(s),
                "v13_visit_rank": row.get("v13_visit_rank"),
            }
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root-stop-alpha", type=float, default=3.0)
    parser.add_argument("--selected-pos-weight", type=float, default=30.0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _load_rows(Path(args.train_jsonl))
    grouped = _group_rows(all_rows)
    train_keys, val_keys = _split_keys(grouped, float(args.train_fraction), int(args.seed))
    train_rows = _rows_for_keys(grouped, train_keys)
    val_rows = _rows_for_keys(grouped, val_keys)
    holdout_rows = _load_rows(Path(args.holdout_jsonl)) if args.holdout_jsonl else []

    names = _feature_names(all_rows)
    x_train_raw = torch.tensor([[float((row.get("features") or {}).get(name, 0.0)) for name in names] for row in train_rows], dtype=torch.float32)
    mean_t = x_train_raw.mean(dim=0)
    std_t = x_train_raw.std(dim=0).clamp_min(1e-6)
    x_train, y_ref_train, y_reg_train, w_train, sel_train, bad_train = _tensorize(train_rows, names, mean_t=mean_t, std_t=std_t)
    x_val, _y_ref_val, _y_reg_val, _w_val, _sel_val, _bad_val = _tensorize(val_rows, names, mean_t=mean_t, std_t=std_t)
    if holdout_rows:
        x_hold, _y_ref_hold, _y_reg_hold, _w_hold, _sel_hold, _bad_hold = _tensorize(holdout_rows, names, mean_t=mean_t, std_t=std_t)
    else:
        x_hold = torch.empty((0, len(names)), dtype=torch.float32)

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model = SidecarMLP(in_dim=len(names), hidden_dim=int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    ds = TensorDataset(x_train, y_ref_train, y_reg_train, w_train, sel_train, bad_train)
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, generator=torch.Generator().manual_seed(int(args.seed)))

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb, rb, wb, sb, root_bad_b in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            rb = rb.to(device)
            wb = wb.to(device)
            sb = sb.to(device)
            root_bad_b = root_bad_b.to(device)
            ref_logit, reg_logit, stop_logit = model(xb)
            bce = nn.functional.binary_cross_entropy_with_logits(ref_logit, yb, reduction="none")
            reg = nn.functional.smooth_l1_loss(torch.sigmoid(reg_logit), rb, reduction="none")
            candidate_loss = ((bce + 0.7 * reg) * wb).sum() / wb.sum().clamp_min(1e-6)
            selected_mask = sb > 0.5
            if bool(selected_mask.any()):
                stop_target = root_bad_b[selected_mask]
                stop_raw = nn.functional.binary_cross_entropy_with_logits(
                    stop_logit[selected_mask],
                    stop_target,
                    reduction="none",
                )
                stop_weight = torch.where(
                    stop_target > 0.5,
                    torch.full_like(stop_target, float(args.selected_pos_weight)),
                    torch.ones_like(stop_target),
                )
                stop_loss = (stop_raw * stop_weight).sum() / stop_weight.sum().clamp_min(1e-6)
            else:
                stop_loss = torch.zeros((), device=device)
            loss = candidate_loss + float(args.root_stop_alpha) * stop_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss.item()) * int(xb.shape[0])
            total_rows += int(xb.shape[0])
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 10) == 0:
            history.append({"epoch": float(epoch), "train_loss": total_loss / max(total_rows, 1)})

    train_probs, train_regs, train_stops = _predict(model, x_train, device, int(args.batch_size))
    val_probs, val_regs, val_stops = _predict(model, x_val, device, int(args.batch_size))
    hold_probs, hold_regs, hold_stops = (
        _predict(model, x_hold, device, int(args.batch_size))
        if holdout_rows
        else (torch.empty(0), torch.empty(0), torch.empty(0))
    )

    ckpt = {
        "model_state_dict": model.cpu().state_dict(),
        "feature_names": names,
        "mean": mean_t,
        "std": std_t,
        "hidden_dim": int(args.hidden_dim),
        "splits": {
            "train_keys": sorted(train_keys),
            "val_keys": sorted(val_keys),
            "train_jsonl": str(Path(args.train_jsonl)),
            "holdout_jsonl": str(Path(args.holdout_jsonl)) if args.holdout_jsonl else "",
        },
        "history": history,
        "root_stop_alpha": float(args.root_stop_alpha),
        "selected_pos_weight": float(args.selected_pos_weight),
    }
    model_path = out_dir / "sidecar.pt"
    torch.save(ckpt, model_path)

    _write_predictions(out_dir / "predictions_train.jsonl", train_rows, train_probs, train_regs, train_stops)
    _write_predictions(out_dir / "predictions_val.jsonl", val_rows, val_probs, val_regs, val_stops)
    if holdout_rows:
        _write_predictions(out_dir / "predictions_holdout.jsonl", holdout_rows, hold_probs, hold_regs, hold_stops)

    summary = {
        "model_path": str(model_path),
        "feature_count": len(names),
        "train": _metrics(train_rows, train_probs, train_regs, train_stops),
        "val": _metrics(val_rows, val_probs, val_regs, val_stops),
        "holdout": _metrics(holdout_rows, hold_probs, hold_regs, hold_stops) if holdout_rows else None,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
