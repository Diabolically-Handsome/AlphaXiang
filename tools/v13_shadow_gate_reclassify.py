#!/usr/bin/env python3
"""Reclassify recorded V13.5 shadow-gate verifier attempts.

This is cheaper and more exact than a fresh offline replay when an
external_arena JSON already contains ``shadow_disagreement_verifier_attempt``
rows with verified child evals.  It lets us sweep split-threshold gate rules
without rerunning Pikafish or changing the arena trajectory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _input_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("external_arena_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("no external_arena_*.json files found")
    return files


def _score(result: str) -> float:
    if result == "our_win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _attempt_rows(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_idx, game in enumerate(payload.get("per_game", []) or [], start=1):
        for stat in game.get("search_stats", []) or []:
            probe = stat.get("shadow_value_probe") or {}
            attempt = probe.get("shadow_disagreement_verifier_attempt") or {}
            if not isinstance(attempt, dict) or not bool(attempt.get("attempted")):
                continue
            rows.append(
                {
                    "source_json": str(source),
                    "game_index": int(game.get("index", game_idx - 1)),
                    "display_game": int(game_idx),
                    "opening_id": str(game.get("opening_id", "")),
                    "ply": int(stat.get("ply", -1)),
                    "side": str(stat.get("side", "")),
                    "result": str(game.get("result", "")),
                    "score": _score(str(game.get("result", ""))),
                    "termination": str(game.get("termination", "")),
                    "actual_move_uci": str(attempt.get("original_move_uci", probe.get("actual_move_uci", ""))),
                    "shadow_best_move_uci": str(attempt.get("shadow_best_move_uci", probe.get("shadow_best_move_uci", ""))),
                    "online_reason": str(attempt.get("reason", "")),
                    "online_accepted": bool(attempt.get("accepted")),
                    "online_improvement_cp": attempt.get("improvement_cp"),
                    "candidates": list(attempt.get("candidates", []) or []),
                }
            )
    return rows


def _classify_row(
    row: dict[str, Any],
    *,
    margin_cp: int,
    mate_risk_margin_cp: int,
    mate_risk_cp: int,
    escape_margin_cp: int,
    escape_risk_cp: int,
    escape_safe_cp: int,
) -> dict[str, Any]:
    actual = str(row.get("actual_move_uci", ""))
    candidates = [
        cand for cand in row.get("candidates", [])
        if cand.get("child_eval_cp_opponent_pov") is not None and not cand.get("illegal")
    ]
    original = next((cand for cand in candidates if str(cand.get("move_uci")) == actual), None)
    best = min(candidates, key=lambda cand: int(cand["child_eval_cp_opponent_pov"]), default=None)
    out = dict(row)
    out.pop("candidates", None)
    if original is None:
        out.update({"accepted": False, "reason": "missing_original_eval"})
        return out
    if best is None:
        out.update({"accepted": False, "reason": "missing_best_eval"})
        return out

    original_cp = int(original["child_eval_cp_opponent_pov"])
    best_cp = int(best["child_eval_cp_opponent_pov"])
    improvement_cp = original_cp - best_cp
    original_mate_risk = original.get("mate_in") is not None or original_cp >= int(mate_risk_cp)
    replacement_mate_risk = best.get("mate_in") is not None or best_cp >= int(mate_risk_cp)
    ordinary_accept = improvement_cp >= int(margin_cp)
    mate_risk_accept = (
        int(mate_risk_margin_cp) >= 0
        and bool(original_mate_risk)
        and not bool(replacement_mate_risk)
        and improvement_cp >= int(mate_risk_margin_cp)
    )
    escape_accept = (
        int(escape_margin_cp) >= 0
        and original_cp >= int(escape_risk_cp)
        and best_cp <= int(escape_safe_cp)
        and improvement_cp >= int(escape_margin_cp)
    )

    if str(best.get("move_uci")) == actual:
        accepted = False
        reason = "verified_original_best"
        acceptance_rule = ""
    elif ordinary_accept or mate_risk_accept or escape_accept:
        accepted = True
        if mate_risk_accept and not ordinary_accept:
            acceptance_rule = "mate_risk"
        elif escape_accept and not ordinary_accept:
            acceptance_rule = "escape"
        else:
            acceptance_rule = "ordinary"
        reason = "accepted"
    else:
        accepted = False
        reason = "improvement_below_margin"
        acceptance_rule = ""

    out.update(
        {
            "accepted": bool(accepted),
            "reason": reason,
            "acceptance_rule": acceptance_rule,
            "best_move_uci": str(best.get("move_uci", "")),
            "improvement_cp": int(improvement_cp),
            "original_child_eval_cp_opponent_pov": int(original_cp),
            "best_child_eval_cp_opponent_pov": int(best_cp),
            "original_mate_in": original.get("mate_in"),
            "best_mate_in": best.get("mate_in"),
            "original_mate_risk": bool(original_mate_risk),
            "replacement_mate_risk": bool(replacement_mate_risk),
            "ordinary_accept": bool(ordinary_accept),
            "mate_risk_accept": bool(mate_risk_accept),
            "escape_accept": bool(escape_accept),
        }
    )
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    return {
        "attempts": len(rows),
        "accepted": len(accepted),
        "accepted_rate": 0.0 if not rows else len(accepted) / len(rows),
        "reason_counts": dict(Counter(str(row.get("reason", "")) for row in rows)),
        "accepted_rules": dict(Counter(str(row.get("acceptance_rule", "")) for row in accepted)),
        "accepted_by_result": dict(Counter(str(row.get("result", "")) for row in accepted)),
        "accepted_by_termination": dict(Counter(str(row.get("termination", "")) for row in accepted)),
        "accepted_by_game": dict(Counter(str(row.get("display_game", "")) for row in accepted)),
        "accepted_score_delta_upper_bound": sum(1.0 - float(row.get("score", 0.0)) for row in accepted),
        "online_accept_match": {
            "same_accept": sum(1 for row in rows if bool(row.get("accepted")) == bool(row.get("online_accepted"))),
            "new_accepts": sum(1 for row in rows if bool(row.get("accepted")) and not bool(row.get("online_accepted"))),
            "dropped_online_accepts": sum(1 for row in rows if not bool(row.get("accepted")) and bool(row.get("online_accepted"))),
        },
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# V13.5 Shadow Gate Reclassification",
        "",
        f"- files: {payload['files']}",
        f"- attempts: {summary['attempts']}",
        f"- accepted: {summary['accepted']} ({100.0 * summary['accepted_rate']:.1f}%)",
        f"- config: {payload['config']}",
        f"- accepted rules: {summary['accepted_rules']}",
        f"- accepted by result: {summary['accepted_by_result']}",
        f"- accepted by termination: {summary['accepted_by_termination']}",
        f"- online accept match: {summary['online_accept_match']}",
        "",
        "## Accepted Rows",
        "",
        "| game | ply | result | term | rule | actual | replacement | improvement | original cp | replacement cp | online reason | opening |",
        "|---:|---:|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        [r for r in payload["rows"] if r.get("accepted")],
        key=lambda r: (int(r.get("display_game", 0)), int(r.get("ply", 0))),
    ):
        lines.append(
            f"| {row.get('display_game')} | {row.get('ply')} | {row.get('result')} | "
            f"{row.get('termination')} | {row.get('acceptance_rule')} | "
            f"{row.get('actual_move_uci')} | {row.get('best_move_uci')} | "
            f"{row.get('improvement_cp')} | {row.get('original_child_eval_cp_opponent_pov')} | "
            f"{row.get('best_child_eval_cp_opponent_pov')} | {row.get('online_reason')} | "
            f"{row.get('opening_id')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_or_dir", nargs="+")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--margin-cp", type=int, default=600)
    parser.add_argument("--mate-risk-margin-cp", type=int, default=300)
    parser.add_argument("--mate-risk-cp", type=int, default=19000)
    parser.add_argument("--escape-margin-cp", type=int, default=450)
    parser.add_argument("--escape-risk-cp", type=int, default=500)
    parser.add_argument("--escape-safe-cp", type=int, default=100)
    args = parser.parse_args()

    files = _input_files(list(args.json_or_dir))
    raw_rows: list[dict[str, Any]] = []
    for path in files:
        raw_rows.extend(_attempt_rows(json.loads(path.read_text(encoding="utf-8")), path))
    rows = [
        _classify_row(
            row,
            margin_cp=int(args.margin_cp),
            mate_risk_margin_cp=int(args.mate_risk_margin_cp),
            mate_risk_cp=int(args.mate_risk_cp),
            escape_margin_cp=int(args.escape_margin_cp),
            escape_risk_cp=int(args.escape_risk_cp),
            escape_safe_cp=int(args.escape_safe_cp),
        )
        for row in raw_rows
    ]
    payload = {
        "files": len(files),
        "config": {
            "margin_cp": int(args.margin_cp),
            "mate_risk_margin_cp": int(args.mate_risk_margin_cp),
            "mate_risk_cp": int(args.mate_risk_cp),
            "escape_margin_cp": int(args.escape_margin_cp),
            "escape_risk_cp": int(args.escape_risk_cp),
            "escape_safe_cp": int(args.escape_safe_cp),
        },
        "summary": _summarize(rows),
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
