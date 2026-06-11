#!/usr/bin/env python3
"""Decision gate for V13.5 shadow-disagreement verifier experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decide(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    baseline = payload.get("baseline_summary", {})
    gated = payload.get("gated_summary", {})
    paired = payload.get("paired_summary", {})
    paired_games = _int(paired.get("paired_games"), 0)
    score_delta = _float(paired.get("score_delta"), 0.0)
    baseline_mate = _int(baseline.get("mate_losses"), 0)
    gated_mate = _int(gated.get("mate_losses"), 0)
    baseline_longcheck = _int(baseline.get("longcheck_losses"), 0)
    gated_longcheck = _int(gated.get("longcheck_losses"), 0)
    gate_events = _int(gated.get("shadow_gate_events"), 0)
    worsened = _int(paired.get("worsened_games"), 0)

    reasons: list[str] = []
    if paired_games < int(args.min_paired_games):
        reasons.append(f"paired games {paired_games} < required {int(args.min_paired_games)}")
    if score_delta < float(args.min_score_delta):
        reasons.append(f"score delta {score_delta:+.3f} < required {float(args.min_score_delta):+.3f}")
    if gated_mate > baseline_mate + int(args.max_mate_loss_increase):
        reasons.append(
            f"mate losses increased from {baseline_mate} to {gated_mate} "
            f"(allowed +{int(args.max_mate_loss_increase)})"
        )
    if gated_longcheck > baseline_longcheck + int(args.max_longcheck_loss_increase):
        reasons.append(
            f"longcheck losses increased from {baseline_longcheck} to {gated_longcheck} "
            f"(allowed +{int(args.max_longcheck_loss_increase)})"
        )
    if int(args.min_gate_events) > 0 and gate_events < int(args.min_gate_events):
        reasons.append(f"shadow gate events {gate_events} < required {int(args.min_gate_events)}")
    if worsened > int(args.max_worsened_games):
        reasons.append(f"worsened paired games {worsened} > allowed {int(args.max_worsened_games)}")

    status = "pass" if not reasons else "fail"
    if paired_games <= 0:
        status = "insufficient"
    return {
        "status": status,
        "passed": status == "pass",
        "reasons": reasons,
        "thresholds": {
            "min_paired_games": int(args.min_paired_games),
            "min_score_delta": float(args.min_score_delta),
            "max_mate_loss_increase": int(args.max_mate_loss_increase),
            "max_longcheck_loss_increase": int(args.max_longcheck_loss_increase),
            "min_gate_events": int(args.min_gate_events),
            "max_worsened_games": int(args.max_worsened_games),
        },
        "observed": {
            "paired_games": paired_games,
            "score_delta": score_delta,
            "baseline_score_rate": baseline.get("score_rate"),
            "gated_score_rate": gated.get("score_rate"),
            "baseline_mate_losses": baseline_mate,
            "gated_mate_losses": gated_mate,
            "baseline_longcheck_losses": baseline_longcheck,
            "gated_longcheck_losses": gated_longcheck,
            "shadow_gate_events": gate_events,
            "improved_games": _int(paired.get("improved_games"), 0),
            "worsened_games": worsened,
            "unchanged_games": _int(paired.get("unchanged_games"), 0),
        },
    }


def _write_md(path: Path, decision: dict[str, Any]) -> None:
    observed = decision["observed"]
    lines = [
        "# V13.5 Shadow Gate Decision",
        "",
        f"- status: **{decision['status']}**",
        f"- passed: {decision['passed']}",
        "",
        "## Observed",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| paired games | {observed['paired_games']} |",
        f"| score delta | {observed['score_delta']:+.3f} |",
        f"| baseline score rate | {observed['baseline_score_rate']} |",
        f"| gated score rate | {observed['gated_score_rate']} |",
        f"| mate losses | {observed['baseline_mate_losses']} -> {observed['gated_mate_losses']} |",
        f"| longcheck losses | {observed['baseline_longcheck_losses']} -> {observed['gated_longcheck_losses']} |",
        f"| shadow gate events | {observed['shadow_gate_events']} |",
        f"| improved / worsened / unchanged | {observed['improved_games']} / {observed['worsened_games']} / {observed['unchanged_games']} |",
    ]
    if decision["reasons"]:
        lines += ["", "## Reasons", ""]
        lines.extend(f"- {reason}" for reason in decision["reasons"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("compare_json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--min-paired-games", type=int, default=12)
    parser.add_argument("--min-score-delta", type=float, default=0.0)
    parser.add_argument("--max-mate-loss-increase", type=int, default=0)
    parser.add_argument("--max-longcheck-loss-increase", type=int, default=0)
    parser.add_argument("--min-gate-events", type=int, default=1)
    parser.add_argument("--max-worsened-games", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.compare_json).read_text(encoding="utf-8"))
    decision = _decide(payload, args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if decision["status"] == "pass":
        return 0
    if decision["status"] == "insufficient":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
