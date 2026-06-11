#!/usr/bin/env python3
"""Offline conservative root gate evaluation for the V13.5 sidecar."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from torch import nn


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


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source_jsonl"] = str(path)
                    rows.append(row)
    return rows


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["root_key"])].append(row)
    return out


def _selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((row for row in rows if bool(row.get("candidate_is_selected"))), rows[0])


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _rank(value: Any, default: int = 999) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    names = list(ckpt["feature_names"])
    mean_t = ckpt["mean"].float()
    std_t = ckpt["std"].float().clamp_min(1e-6)
    model = SidecarMLP(len(names), int(ckpt.get("hidden_dim", 64)))
    state = ckpt["model_state_dict"]
    if "stop_head.weight" not in state:
        # Backward compatibility for early two-head smoke checkpoints.
        model.stop_head.load_state_dict(model.refute_head.state_dict())
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, names, mean_t, std_t, ckpt


@torch.no_grad()
def _predict_rows(rows: list[dict[str, Any]], model: SidecarMLP, names: list[str], mean_t: torch.Tensor, std_t: torch.Tensor, device: torch.device, batch_size: int) -> None:
    xs = []
    for row in rows:
        feats = row.get("features") or {}
        xs.append([float(feats.get(name, 0.0)) for name in names])
    x = (torch.tensor(xs, dtype=torch.float32) - mean_t) / std_t
    probs: list[torch.Tensor] = []
    regs: list[torch.Tensor] = []
    stops: list[torch.Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device)
        ref_logit, reg_logit, stop_logit = model(xb)
        probs.append(torch.sigmoid(ref_logit).cpu())
        regs.append(torch.sigmoid(reg_logit).cpu())
        stops.append(torch.sigmoid(stop_logit).cpu())
    p = torch.cat(probs).tolist()
    r = torch.cat(regs).tolist()
    s = torch.cat(stops).tolist()
    for row, pp, rr, ss in zip(rows, p, r, s):
        row["pred_refute_prob"] = float(pp)
        row["pred_regret_log01"] = float(rr)
        row["pred_stop_prob"] = float(ss)


def _candidate_pool(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    pool = [
        row for row in rows
        if not bool(row.get("candidate_is_selected"))
        and _rank(row.get("v13_visit_rank")) <= int(top_k)
    ]
    pool.sort(key=lambda row: (_rank(row.get("v13_visit_rank")), -_float(row.get("pred_refute_prob"), 0.0)))
    return pool


def _eval_one_root(rows: list[dict[str, Any]], threshold: float, top_k: int, override_margin_cp: float, mate_margin_cp: float) -> dict[str, Any]:
    sel = _selected(rows)
    sel_score = _float(sel.get("candidate_child_d16_score_cp"), 0.0)
    sel_regret = _float(sel.get("selected_regret_cp"), _float(sel.get("candidate_regret_cp"), 0.0))
    root_bad = bool(sel.get("root_selected_bad"))
    root_cat = bool(sel.get("root_catastrophic"))
    selected_risk = _float(sel.get("pred_stop_prob", sel.get("pred_refute_prob")), 0.0)
    triggered = selected_risk >= float(threshold)
    selected_mate = sel.get("candidate_mate_in_child") is not None
    replacement = None
    if triggered:
        for cand in sorted(_candidate_pool(rows, top_k), key=lambda row: _float(row.get("candidate_child_d16_score_cp"), -1e9), reverse=True):
            margin = _float(cand.get("candidate_child_d16_score_cp"), -1e9) - sel_score
            required = float(mate_margin_cp) if selected_mate else float(override_margin_cp)
            if margin >= required:
                replacement = cand
                break
    replacement_regret = None if replacement is None else _float(replacement.get("candidate_regret_cp"), 0.0)
    prevented = bool(root_bad and replacement is not None and replacement_regret is not None and replacement_regret <= max(sel_regret - 100.0, 0.0))
    regression = bool(
        (not root_bad)
        and replacement is not None
        and replacement_regret is not None
        and replacement_regret >= max(150.0, sel_regret + 100.0)
    )
    return {
        "root_key": sel["root_key"],
        "selected_move": sel["selected_move"],
        "teacher_best_move": sel["teacher_best_move"],
        "root_class": sel.get("root_class"),
        "root_bad": root_bad,
        "root_catastrophic": root_cat,
        "selected_regret_cp": sel_regret,
        "selected_score_cp": sel_score,
        "selected_risk": selected_risk,
        "selected_candidate_refute_risk": _float(sel.get("pred_refute_prob"), 0.0),
        "triggered": bool(triggered),
        "replacement_move": None if replacement is None else replacement.get("candidate_move"),
        "replacement_regret_cp": replacement_regret,
        "replacement_score_cp": None if replacement is None else _float(replacement.get("candidate_child_d16_score_cp"), 0.0),
        "prevented": prevented,
        "regression": regression,
        "source_jsonl": sel.get("_source_jsonl"),
    }


def _eval_roots(grouped: dict[str, list[dict[str, Any]]], keys: set[str], threshold: float, top_k: int, override_margin_cp: float, mate_margin_cp: float) -> dict[str, Any]:
    records = [
        _eval_one_root(grouped[key], threshold, top_k, override_margin_cp, mate_margin_cp)
        for key in sorted(keys)
        if key in grouped
    ]
    bad = [r for r in records if r["root_bad"]]
    clean = [r for r in records if not r["root_bad"]]
    cat = [r for r in records if r["root_catastrophic"]]
    regressions = sum(1 for r in clean if r["regression"])
    return {
        "threshold": float(threshold),
        "roots": len(records),
        "bad_roots": len(bad),
        "catastrophic_roots": len(cat),
        "clean_roots": len(clean),
        "triggered_roots": sum(1 for r in records if r["triggered"]),
        "triggered_bad_roots": sum(1 for r in bad if r["triggered"]),
        "triggered_catastrophic_roots": sum(1 for r in cat if r["triggered"]),
        "triggered_clean_roots": sum(1 for r in clean if r["triggered"]),
        "bad_prevented": sum(1 for r in bad if r["prevented"]),
        "catastrophic_prevented": sum(1 for r in cat if r["prevented"]),
        "new_non_bad_regressions": regressions,
        "new_non_bad_regression_rate_pct": 0.0 if not clean else 100.0 * regressions / len(clean),
        "selected_risk_median": None if not records else median([r["selected_risk"] for r in records]),
        "records": records,
    }


def _choose_threshold(grouped: dict[str, list[dict[str, Any]]], val_keys: set[str], top_k: int, override_margin_cp: float, mate_margin_cp: float, max_regression_pct: float) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    grid = [x / 100.0 for x in range(5, 96, 5)]
    trials = [
        _eval_roots(grouped, val_keys, th, top_k, override_margin_cp, mate_margin_cp)
        for th in grid
    ]
    passing = [
        trial for trial in trials
        if trial["new_non_bad_regression_rate_pct"] <= float(max_regression_pct)
    ]
    if passing:
        chosen = max(
            passing,
            key=lambda t: (
                t["bad_prevented"],
                t["catastrophic_prevented"],
                t["triggered_bad_roots"],
                t["triggered_catastrophic_roots"],
                -t["triggered_clean_roots"],
                -t["threshold"],
            ),
        )
    else:
        chosen = max(
            trials,
            key=lambda t: (
                -t["new_non_bad_regression_rate_pct"],
                t["bad_prevented"],
                t["catastrophic_prevented"],
            ),
        )
    return float(chosen["threshold"]), chosen, trials


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    combined = summary["combined"]
    exact = summary.get("exact_holdout")
    lines = [
        "# V13.5 Root-Q Sidecar Gate Eval",
        "",
        f"- threshold: {summary['threshold']:.2f}",
        f"- combined bad prevented: {combined['bad_prevented']}/{combined['bad_roots']}",
        f"- combined catastrophic prevented: {combined['catastrophic_prevented']}/{combined['catastrophic_roots']}",
        f"- combined new clean regressions: {combined['new_non_bad_regressions']}/{combined['clean_roots']} ({combined['new_non_bad_regression_rate_pct']:.2f}%)",
    ]
    if exact is not None:
        lines.append(f"- exact holdout new regressions: {exact['new_non_bad_regressions']}/{exact['clean_roots']} ({exact['new_non_bad_regression_rate_pct']:.2f}%)")
    lines += [
        f"- pass: {summary['passed']}",
        "",
        "## Worst Triggered Bad Roots",
        "",
        "| selected regret | selected | replacement | class | prevented | risk |",
        "|---:|---|---|---|---|---:|",
    ]
    bad_records = [r for r in combined["records"] if r["root_bad"] and r["triggered"]]
    for rec in sorted(bad_records, key=lambda r: r["selected_regret_cp"], reverse=True)[:20]:
        lines.append(
            f"| {rec['selected_regret_cp']:.1f} | {rec['selected_move']} | {rec['replacement_move']} | "
            f"{rec['root_class']} | {rec['prevented']} | {rec['selected_risk']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--expanded-jsonl", required=True)
    parser.add_argument("--exact-jsonl", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--threshold-json", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--override-margin-cp", type=float, default=300.0)
    parser.add_argument("--mate-margin-cp", type=float, default=100.0)
    parser.add_argument("--max-new-regression-pct", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, names, mean_t, std_t, ckpt = _load_model(Path(args.sidecar), device)
    expanded_rows = _load_rows([Path(args.expanded_jsonl)])
    exact_rows = _load_rows([Path(args.exact_jsonl)]) if args.exact_jsonl else []
    all_rows = expanded_rows + exact_rows
    _predict_rows(all_rows, model, names, mean_t, std_t, device, int(args.batch_size))
    grouped = _group(all_rows)
    expanded_grouped = _group(expanded_rows)
    exact_grouped = _group(exact_rows)
    val_keys = set(ckpt.get("splits", {}).get("val_keys", []))
    if not val_keys:
        val_keys = set(expanded_grouped)

    threshold, threshold_eval, threshold_trials = _choose_threshold(
        grouped,
        val_keys,
        int(args.top_k),
        float(args.override_margin_cp),
        float(args.mate_margin_cp),
        float(args.max_new_regression_pct),
    )
    combined_keys = set(grouped)
    expanded_keys = set(expanded_grouped)
    exact_keys = set(exact_grouped)
    combined = _eval_roots(grouped, combined_keys, threshold, int(args.top_k), float(args.override_margin_cp), float(args.mate_margin_cp))
    expanded = _eval_roots(grouped, expanded_keys, threshold, int(args.top_k), float(args.override_margin_cp), float(args.mate_margin_cp))
    exact = _eval_roots(grouped, exact_keys, threshold, int(args.top_k), float(args.override_margin_cp), float(args.mate_margin_cp)) if exact_keys else None

    passed = bool(
        combined["bad_prevented"] >= 10
        and combined["catastrophic_prevented"] >= 5
        and combined["new_non_bad_regression_rate_pct"] <= 1.5
        and (exact is None or exact["new_non_bad_regression_rate_pct"] <= 2.0)
    )
    summary = {
        "sidecar": str(Path(args.sidecar)),
        "threshold": threshold,
        "threshold_selected_on_val": {k: v for k, v in threshold_eval.items() if k != "records"},
        "threshold_trials": [{k: v for k, v in t.items() if k != "records"} for t in threshold_trials],
        "top_k": int(args.top_k),
        "override_margin_cp": float(args.override_margin_cp),
        "mate_margin_cp": float(args.mate_margin_cp),
        "combined": combined,
        "expanded": expanded,
        "exact_holdout": exact,
        "passed": passed,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.threshold_json:
        Path(args.threshold_json).write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "top_k": int(args.top_k),
                    "override_margin_cp": float(args.override_margin_cp),
                    "mate_margin_cp": float(args.mate_margin_cp),
                    "passed": passed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.out_md:
        _write_md(Path(args.out_md), summary)
    printable = {k: v for k, v in summary.items() if k not in {"combined", "expanded", "exact_holdout", "threshold_trials"}}
    printable["combined"] = {k: v for k, v in combined.items() if k != "records"}
    printable["exact_holdout"] = None if exact is None else {k: v for k, v in exact.items() if k != "records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
