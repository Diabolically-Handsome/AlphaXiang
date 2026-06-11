#!/usr/bin/env python3
"""Offline upper-bound eval for scalar/WDL disagreement-triggered verification.

Input is a trajectory audit JSON produced by ``v13_root_trajectory_audit.py``
with both scalar and WDL value sources.  The tool simulates a conservative
policy:

1. Use scalar MCTS as the anchor move.
2. Trigger only when scalar and WDL root best moves disagree.
3. If triggered, send the scalar/WDL top-K union to a hypothetical deep child
   verifier.  Offline, d16 ``known_regret_cp`` is used as the verifier label.
4. Override only when the best labeled candidate improves over scalar by the
   required margin.

This is an upper bound because real arena would have to query the verifier live.
It is still useful because it tells us whether scalar/WDL disagreement contains
enough actionable signal to justify that engineering.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _find_config(search: dict[str, Any], value_source: str) -> tuple[str, dict[str, Any]]:
    for name, cfg in search.items():
        if str(name).startswith(f"{value_source}_"):
            return name, cfg
    for name, cfg in search.items():
        if str(cfg.get("value_source", "")) == value_source:
            return name, cfg
    fallback = f"{value_source}_baseline"
    if fallback in search:
        return fallback, search[fallback]
    raise KeyError(f"could not find value source config: {value_source}")


def _known(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    return _float(row.get("known_regret_cp"), None)


def _top_union(*finals: dict[str, Any], top_k: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for final in finals:
        for row in (final.get("top_moves") or [])[: max(1, int(top_k))]:
            move = str(row.get("move_uci", ""))[:4]
            if not move:
                continue
            prev = out.get(move)
            regret = _known(row)
            if prev is None:
                out[move] = dict(row)
                continue
            prev_regret = _known(prev)
            if prev_regret is None or (regret is not None and regret < prev_regret):
                merged = dict(prev)
                merged.update(row)
                out[move] = merged
    return out


def _best_labeled_candidate(candidates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    labeled = [row for row in candidates.values() if _known(row) is not None]
    if not labeled:
        return None
    return min(labeled, key=lambda row: float(_known(row) or 1e9))


def _eval_record(record: dict[str, Any], *, top_k: int, override_margin_cp: float, require_wdl_candidate: bool) -> dict[str, Any]:
    root = record.get("root") or {}
    scalar_name, scalar_cfg = _find_config(record.get("search") or {}, "scalar")
    wdl_name, wdl_cfg = _find_config(record.get("search") or {}, "wdl")
    scalar_final = scalar_cfg.get("final") or {}
    wdl_final = wdl_cfg.get("final") or {}

    scalar_move = str(scalar_final.get("best_move", ""))[:4]
    wdl_move = str(wdl_final.get("best_move", ""))[:4]
    scalar_regret = _float(scalar_final.get("best_move_known_regret_cp"), None)
    wdl_regret = _float(wdl_final.get("best_move_known_regret_cp"), None)
    selected_regret = _float(root.get("selected_regret_cp"), 0.0) or 0.0
    root_bad = bool(root.get("is_bad", selected_regret >= 150.0))
    root_cat = bool(root.get("is_catastrophic", root_bad and selected_regret >= 1000.0))
    disagrees = bool(scalar_move and wdl_move and scalar_move != wdl_move)
    candidates = _top_union(scalar_final, wdl_final, top_k=int(top_k))
    best = _best_labeled_candidate(candidates)
    best_regret = _known(best)
    best_move = None if best is None else str(best.get("move_uci", ""))[:4]
    triggered = bool(disagrees)
    if require_wdl_candidate and best_move != wdl_move:
        triggered = False

    override = False
    if triggered and scalar_regret is not None and best_regret is not None and best_move and best_move != scalar_move:
        override = bool(float(scalar_regret) - float(best_regret) >= float(override_margin_cp))

    final_move = best_move if override else scalar_move
    final_regret = best_regret if override else scalar_regret
    scalar_bad = bool(scalar_regret is not None and scalar_regret >= 150.0)
    final_bad = bool(final_regret is not None and final_regret >= 150.0)
    scalar_safe = bool(scalar_regret is not None and scalar_regret < 150.0)
    final_safe = bool(final_regret is not None and final_regret < 150.0)
    selected_safe = selected_regret < 150.0
    baseline_regressed_clean = bool((not root_bad) and scalar_regret is not None and scalar_regret >= max(150.0, selected_regret + 100.0))
    gate_regressed_clean = bool((not root_bad) and final_regret is not None and final_regret >= max(150.0, selected_regret + 100.0))

    return {
        "root_key": root.get("key", ""),
        "classification": record.get("classification", "unknown"),
        "termination": root.get("termination", ""),
        "selected_move": root.get("selected_move", ""),
        "teacher_best_move": root.get("teacher_best_move", ""),
        "selected_regret_cp": selected_regret,
        "root_bad": root_bad,
        "root_catastrophic": root_cat,
        "scalar_config": scalar_name,
        "wdl_config": wdl_name,
        "scalar_move": scalar_move,
        "scalar_regret_cp": scalar_regret,
        "wdl_move": wdl_move,
        "wdl_regret_cp": wdl_regret,
        "disagrees": disagrees,
        "triggered": triggered,
        "candidate_count": len(candidates),
        "labeled_candidate_count": sum(1 for row in candidates.values() if _known(row) is not None),
        "verifier_best_move": best_move,
        "verifier_best_regret_cp": best_regret,
        "override": override,
        "final_move": final_move,
        "final_regret_cp": final_regret,
        "root_selected_safe": selected_safe,
        "scalar_safe": scalar_safe,
        "final_safe": final_safe,
        "scalar_bad": scalar_bad,
        "final_bad": final_bad,
        "bad_prevented_vs_scalar": bool(root_bad and scalar_bad and final_safe),
        "bad_prevented_vs_original": bool(root_bad and final_safe),
        "bad_improved_vs_scalar": bool(root_bad and scalar_regret is not None and final_regret is not None and final_regret <= max(float(scalar_regret) - 100.0, 0.0)),
        "catastrophic_prevented_vs_scalar": bool(root_cat and scalar_bad and final_safe),
        "catastrophic_prevented_vs_original": bool(root_cat and final_safe),
        "baseline_clean_regression": baseline_regressed_clean,
        "gate_clean_regression": gate_regressed_clean,
        "new_clean_regression": bool((not baseline_regressed_clean) and gate_regressed_clean),
        "clean_regression_fixed": bool(baseline_regressed_clean and not gate_regressed_clean),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = [r for r in rows if r["root_bad"]]
    clean = [r for r in rows if not r["root_bad"]]
    cat = [r for r in rows if r["root_catastrophic"]]
    triggered = [r for r in rows if r["triggered"]]
    override = [r for r in rows if r["override"]]
    scalar_bad = [r for r in bad if r["scalar_bad"]]
    final_bad = [r for r in bad if r["final_bad"]]
    scalar_clean_reg = [r for r in clean if r["baseline_clean_regression"]]
    gate_clean_reg = [r for r in clean if r["gate_clean_regression"]]
    labeled_counts = [int(r["labeled_candidate_count"]) for r in rows]
    return {
        "roots": len(rows),
        "bad_roots": len(bad),
        "catastrophic_roots": len(cat),
        "clean_roots": len(clean),
        "triggered_roots": len(triggered),
        "triggered_bad_roots": sum(1 for r in triggered if r["root_bad"]),
        "triggered_clean_roots": sum(1 for r in triggered if not r["root_bad"]),
        "override_roots": len(override),
        "override_bad_roots": sum(1 for r in override if r["root_bad"]),
        "override_clean_roots": sum(1 for r in override if not r["root_bad"]),
        "scalar_bad_roots_after_search": len(scalar_bad),
        "gate_bad_roots_after_search": len(final_bad),
        "scalar_bad_root_safe_rate": 0.0 if not bad else sum(1 for r in bad if r["scalar_safe"]) / len(bad),
        "gate_bad_root_safe_rate": 0.0 if not bad else sum(1 for r in bad if r["final_safe"]) / len(bad),
        "bad_prevented_vs_scalar": sum(1 for r in bad if r["bad_prevented_vs_scalar"]),
        "bad_prevented_vs_original": sum(1 for r in bad if r["bad_prevented_vs_original"]),
        "bad_improved_vs_scalar": sum(1 for r in bad if r["bad_improved_vs_scalar"]),
        "catastrophic_prevented_vs_scalar": sum(1 for r in cat if r["catastrophic_prevented_vs_scalar"]),
        "catastrophic_prevented_vs_original": sum(1 for r in cat if r["catastrophic_prevented_vs_original"]),
        "baseline_clean_regressions": len(scalar_clean_reg),
        "gate_clean_regressions": len(gate_clean_reg),
        "new_clean_regressions": sum(1 for r in clean if r["new_clean_regression"]),
        "clean_regressions_fixed": sum(1 for r in clean if r["clean_regression_fixed"]),
        "mean_labeled_candidates": None if not labeled_counts else mean(labeled_counts),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# V13.5 Scalar/WDL Disagreement Gate Eval",
        "",
        f"- roots: {s['roots']}",
        f"- bad roots: {s['bad_roots']}",
        f"- catastrophic roots: {s['catastrophic_roots']}",
        f"- clean roots: {s['clean_roots']}",
        f"- triggered roots: {s['triggered_roots']} ({s['triggered_bad_roots']} bad / {s['triggered_clean_roots']} clean)",
        f"- overrides: {s['override_roots']} ({s['override_bad_roots']} bad / {s['override_clean_roots']} clean)",
        "",
        "## Gate Impact",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| scalar safe bad roots | {s['scalar_bad_root_safe_rate'] * 100.0:.1f}% |",
        f"| gate safe bad roots | {s['gate_bad_root_safe_rate'] * 100.0:.1f}% |",
        f"| bad prevented vs scalar | {s['bad_prevented_vs_scalar']}/{s['bad_roots']} |",
        f"| bad safe after gate vs original | {s['bad_prevented_vs_original']}/{s['bad_roots']} |",
        f"| catastrophic prevented vs scalar | {s['catastrophic_prevented_vs_scalar']}/{s['catastrophic_roots']} |",
        f"| catastrophic safe after gate vs original | {s['catastrophic_prevented_vs_original']}/{s['catastrophic_roots']} |",
        f"| baseline clean regressions | {s['baseline_clean_regressions']}/{s['clean_roots']} |",
        f"| gate clean regressions | {s['gate_clean_regressions']}/{s['clean_roots']} |",
        f"| new clean regressions | {s['new_clean_regressions']}/{s['clean_roots']} |",
        f"| clean regressions fixed | {s['clean_regressions_fixed']}/{s['clean_roots']} |",
        "",
        "## Overrides",
        "",
        "| regret | class | scalar | scalar regret | verifier | verifier regret | final safe | clean regression |",
        "|---:|---|---|---:|---|---:|---|---|",
    ]
    for row in sorted([r for r in payload["records"] if r["override"]], key=lambda r: (-float(r["selected_regret_cp"]), r["classification"])):
        lines.append(
            f"| {row['selected_regret_cp']:.1f} | {row['classification']} | {row['scalar_move']} | "
            f"{_fmt(row['scalar_regret_cp'])} | {row['verifier_best_move']} | {_fmt(row['verifier_best_regret_cp'])} | "
            f"{row['final_safe']} | {row['gate_clean_regression']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    x = _float(value, None)
    return "n/a" if x is None else f"{x:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--override-margin-cp", type=float, default=300.0)
    parser.add_argument("--require-wdl-candidate", action="store_true",
                        help="Only override if the verifier-best labeled candidate is exactly WDL's root best.")
    args = parser.parse_args()

    payload_in = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    records = [
        _eval_record(
            record,
            top_k=int(args.top_k),
            override_margin_cp=float(args.override_margin_cp),
            require_wdl_candidate=bool(args.require_wdl_candidate),
        )
        for record in payload_in.get("records", [])
    ]
    payload = {
        "trajectory_json": str(Path(args.trajectory_json)),
        "top_k": int(args.top_k),
        "override_margin_cp": float(args.override_margin_cp),
        "require_wdl_candidate": bool(args.require_wdl_candidate),
        "summary": _summarize(records),
        "records": records,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
