#!/usr/bin/env python3
"""Audit shadow value-source probes from external_arena JSON logs.

The arena shadow probe is intentionally non-invasive: it records what a second
MCTS would have chosen with a different value source, but the game still uses
the main search move.  This script turns those per-move shadow rows into a
compact report so we can judge whether scalar/WDL disagreement is a useful
failure-risk signal before adding any gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
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
        raise FileNotFoundError("no arena JSON files found")
    return files


def _result_bucket(result: str) -> str:
    if result == "our_win":
        return "win"
    if result == "opp_win":
        return "loss"
    return "draw"


def _rate(num: int, den: int) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _termination_is_bad(term: str) -> bool:
    return str(term) in {"mate", "longcheck"}


def _probe_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in game.get("search_stats", []) or []:
        probe = event.get("shadow_value_probe")
        if not isinstance(probe, dict) or not probe:
            continue
        attempt = probe.get("shadow_disagreement_verifier_attempt")
        if not isinstance(attempt, dict):
            attempt = {}
        rows.append(
            {
                "game_index": int(game.get("index", event.get("game_index", -1))),
                "ply": int(event.get("ply", -1)),
                "side": str(event.get("side", "")),
                "actual_move_uci": str(probe.get("actual_move_uci", event.get("move_uci", ""))),
                "shadow_best_move_uci": str(probe.get("shadow_best_move_uci", "")),
                "disagrees": bool(probe.get("disagrees_with_actual")),
                "actual_rank_in_shadow_topk": probe.get("actual_rank_in_shadow_topk"),
                "actual_visit_prob_in_shadow_topk": probe.get("actual_visit_prob_in_shadow_topk"),
                "root_value": event.get("root_value"),
                "shadow_root_value": probe.get("shadow_root_value"),
                "shadow_value_source": probe.get("shadow_value_source"),
                "shadow_sims": probe.get("shadow_sims"),
                "shadow_top_moves": probe.get("shadow_top_moves", []),
                "shadow_gate_attempted": bool(attempt.get("attempted")),
                "shadow_gate_accepted": bool(attempt.get("accepted")),
                "shadow_gate_reason": str(attempt.get("reason", "")),
                "shadow_gate_improvement_cp": attempt.get("improvement_cp"),
                "shadow_gate_candidate_count": attempt.get("candidate_count"),
                "shadow_gate_best_move_uci": str(attempt.get("best_move_uci", "")),
            }
        )
    return rows


def _shadow_gate_events(game: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in game.get("guard_events", []) or []:
        if str(event.get("guard_type", "")) != "shadow_value_disagreement_verifier":
            continue
        out.append(
            {
                "game_index": int(game.get("index", event.get("game_index", -1))),
                "ply": int(event.get("ply", -1)),
                "side": str(event.get("side", "")),
                "original_move_uci": str(event.get("original_move_uci", "")),
                "replacement_move_uci": str(event.get("replacement_move_uci", "")),
                "shadow_best_move_uci": str(event.get("shadow_best_move_uci", "")),
                "improvement_cp": event.get("improvement_cp"),
                "margin_cp": event.get("margin_cp"),
            }
        )
    return out


def _summarize_games(games: list[dict[str, Any]], early_ply_threshold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game_records: list[dict[str, Any]] = []
    all_probe_rows: list[dict[str, Any]] = []
    for game in games:
        probes = _probe_rows(game)
        gate_events = _shadow_gate_events(game)
        disagreements = [row for row in probes if row["disagrees"]]
        gate_attempts = [row for row in probes if row["shadow_gate_attempted"]]
        accepted_attempts = [row for row in gate_attempts if row["shadow_gate_accepted"]]
        early_disagreements = [row for row in disagreements if 0 <= int(row["ply"]) <= int(early_ply_threshold)]
        all_probe_rows.extend(probes)
        first = min((int(row["ply"]) for row in disagreements), default=None)
        first_early = min((int(row["ply"]) for row in early_disagreements), default=None)
        first_attempt = min((int(row["ply"]) for row in gate_attempts), default=None)
        game_records.append(
            {
                "game_index": int(game.get("index", -1)),
                "our_side": str(game.get("our_side", "")),
                "result": str(game.get("result", "")),
                "result_bucket": _result_bucket(str(game.get("result", ""))),
                "termination": str(game.get("termination", "")),
                "plies": int(game.get("plies", 0) or 0),
                "opening_id": str(game.get("opening_id", "")),
                "opening_index": game.get("opening_index"),
                "probes": int(len(probes)),
                "disagreements": int(len(disagreements)),
                "disagreement_rate": _rate(len(disagreements), len(probes)),
                "has_disagreement": bool(disagreements),
                "early_disagreements": int(len(early_disagreements)),
                "has_early_disagreement": bool(early_disagreements),
                "shadow_gate_events": int(len(gate_events)),
                "shadow_gate_attempts": int(len(gate_attempts)),
                "shadow_gate_accepted_attempts": int(len(accepted_attempts)),
                "has_shadow_gate_event": bool(gate_events),
                "first_shadow_gate_ply": min((int(row["ply"]) for row in gate_events), default=None),
                "first_shadow_gate_attempt_ply": first_attempt,
                "first_disagreement_ply": first,
                "first_early_disagreement_ply": first_early,
            }
        )

    games_with_probe = [g for g in game_records if g["probes"] > 0]
    with_disagreement = [g for g in games_with_probe if g["has_disagreement"]]
    without_disagreement = [g for g in games_with_probe if not g["has_disagreement"]]
    with_early = [g for g in games_with_probe if g["has_early_disagreement"]]
    without_early = [g for g in games_with_probe if not g["has_early_disagreement"]]
    with_gate = [g for g in games_with_probe if g["has_shadow_gate_event"]]
    without_gate = [g for g in games_with_probe if not g["has_shadow_gate_event"]]
    losses = [g for g in games_with_probe if g["result_bucket"] == "loss"]
    bad_terms = [g for g in games_with_probe if _termination_is_bad(g["termination"])]
    first_plies = [int(g["first_disagreement_ply"]) for g in with_disagreement if g["first_disagreement_ply"] is not None]

    def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "games": len(rows),
            "wins": sum(1 for g in rows if g["result_bucket"] == "win"),
            "losses": sum(1 for g in rows if g["result_bucket"] == "loss"),
            "draws": sum(1 for g in rows if g["result_bucket"] == "draw"),
            "loss_rate": _rate(sum(1 for g in rows if g["result_bucket"] == "loss"), len(rows)),
            "mate_or_longcheck_rate": _rate(sum(1 for g in rows if _termination_is_bad(g["termination"])), len(rows)),
            "avg_disagreement_rate": None if not rows else mean([float(g["disagreement_rate"]) for g in rows]),
        }

    disagree_probe_rows = [row for row in all_probe_rows if row["disagrees"]]
    gate_attempt_rows = [row for row in all_probe_rows if row["shadow_gate_attempted"]]
    gate_improvements = [
        int(row["shadow_gate_improvement_cp"])
        for row in gate_attempt_rows
        if row.get("shadow_gate_improvement_cp") is not None
    ]
    summary = {
        "games": len(game_records),
        "games_with_probe": len(games_with_probe),
        "total_probes": len(all_probe_rows),
        "total_disagreements": len(disagree_probe_rows),
        "disagreement_rate": _rate(len(disagree_probe_rows), len(all_probe_rows)),
        "games_with_disagreement": len(with_disagreement),
        "games_with_early_disagreement": len(with_early),
        "early_ply_threshold": int(early_ply_threshold),
        "loss_games": len(losses),
        "mate_or_longcheck_games": len(bad_terms),
        "first_disagreement_ply": {
            "mean": None if not first_plies else mean(first_plies),
            "median": None if not first_plies else median(first_plies),
            "min": None if not first_plies else min(first_plies),
            "max": None if not first_plies else max(first_plies),
        },
        "by_disagreement": {
            "with": group_metrics(with_disagreement),
            "without": group_metrics(without_disagreement),
        },
        "by_early_disagreement": {
            "with": group_metrics(with_early),
            "without": group_metrics(without_early),
        },
        "by_shadow_gate_event": {
            "with": group_metrics(with_gate),
            "without": group_metrics(without_gate),
        },
        "shadow_gate_games": len(with_gate),
        "shadow_gate_events": sum(int(g["shadow_gate_events"]) for g in games_with_probe),
        "shadow_gate_attempts": len(gate_attempt_rows),
        "shadow_gate_accepted_attempts": sum(1 for row in gate_attempt_rows if row["shadow_gate_accepted"]),
        "shadow_gate_attempt_reasons": dict(Counter(row["shadow_gate_reason"] for row in gate_attempt_rows)),
        "shadow_gate_improvement_cp": {
            "mean": None if not gate_improvements else mean(gate_improvements),
            "median": None if not gate_improvements else median(gate_improvements),
            "min": None if not gate_improvements else min(gate_improvements),
            "max": None if not gate_improvements else max(gate_improvements),
        },
        "disagreements_by_side": dict(Counter(row["side"] for row in disagree_probe_rows)),
        "top_actual_moves_when_disagree": Counter(row["actual_move_uci"] for row in disagree_probe_rows).most_common(12),
        "top_shadow_moves_when_disagree": Counter(row["shadow_best_move_uci"] for row in disagree_probe_rows).most_common(12),
    }
    return summary, game_records


def _audit_files(files: list[Path], early_ply_threshold: int) -> dict[str, Any]:
    all_games: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        games = list(payload.get("per_game", []) or [])
        summary, records = _summarize_games(games, early_ply_threshold)
        file_summaries.append(
            {
                "path": str(path),
                "config": payload.get("config", {}),
                "result": {
                    "score_rate": payload.get("score_rate"),
                    "our_wins": payload.get("our_wins"),
                    "opp_wins": payload.get("opp_wins"),
                    "draws": payload.get("draws"),
                    "termination_counts": payload.get("termination_counts", {}),
                },
                "shadow_value_summary": payload.get("shadow_value_summary", {}),
                "audit_summary": summary,
            }
        )
        for rec in records:
            rec["source_json"] = str(path)
        all_games.extend(records)
    # Recompute combined metrics from game_records directly to keep source_json.
    games_with_probe = [g for g in all_games if g["probes"] > 0]
    with_disagreement = [g for g in games_with_probe if g["has_disagreement"]]
    without_disagreement = [g for g in games_with_probe if not g["has_disagreement"]]
    with_early = [g for g in games_with_probe if g["has_early_disagreement"]]
    without_early = [g for g in games_with_probe if not g["has_early_disagreement"]]
    with_gate = [g for g in games_with_probe if g["has_shadow_gate_event"]]
    without_gate = [g for g in games_with_probe if not g["has_shadow_gate_event"]]
    gate_attempt_rows = [
        row
        for path in files
        for game in (json.loads(path.read_text(encoding="utf-8")).get("per_game", []) or [])
        for row in _probe_rows(game)
        if row["shadow_gate_attempted"]
    ]
    gate_improvements = [
        int(row["shadow_gate_improvement_cp"])
        for row in gate_attempt_rows
        if row.get("shadow_gate_improvement_cp") is not None
    ]

    def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "games": len(rows),
            "wins": sum(1 for g in rows if g["result_bucket"] == "win"),
            "losses": sum(1 for g in rows if g["result_bucket"] == "loss"),
            "draws": sum(1 for g in rows if g["result_bucket"] == "draw"),
            "loss_rate": _rate(sum(1 for g in rows if g["result_bucket"] == "loss"), len(rows)),
            "mate_or_longcheck_rate": _rate(sum(1 for g in rows if _termination_is_bad(g["termination"])), len(rows)),
            "avg_disagreement_rate": None if not rows else mean([float(g["disagreement_rate"]) for g in rows]),
        }

    first_plies = [int(g["first_disagreement_ply"]) for g in with_disagreement if g["first_disagreement_ply"] is not None]
    combined = {
        "files": len(files),
        "games": len(all_games),
        "games_with_probe": len(games_with_probe),
        "total_probes": sum(int(g["probes"]) for g in games_with_probe),
        "total_disagreements": sum(int(g["disagreements"]) for g in games_with_probe),
        "disagreement_rate": _rate(
            sum(int(g["disagreements"]) for g in games_with_probe),
            sum(int(g["probes"]) for g in games_with_probe),
        ),
        "games_with_disagreement": len(with_disagreement),
        "games_with_early_disagreement": len(with_early),
        "early_ply_threshold": int(early_ply_threshold),
        "first_disagreement_ply": {
            "mean": None if not first_plies else mean(first_plies),
            "median": None if not first_plies else median(first_plies),
            "min": None if not first_plies else min(first_plies),
            "max": None if not first_plies else max(first_plies),
        },
        "by_disagreement": {
            "with": group_metrics(with_disagreement),
            "without": group_metrics(without_disagreement),
        },
        "by_early_disagreement": {
            "with": group_metrics(with_early),
            "without": group_metrics(without_early),
        },
        "by_shadow_gate_event": {
            "with": group_metrics(with_gate),
            "without": group_metrics(without_gate),
        },
        "shadow_gate_games": len(with_gate),
        "shadow_gate_events": sum(int(g["shadow_gate_events"]) for g in games_with_probe),
        "shadow_gate_attempts": len(gate_attempt_rows),
        "shadow_gate_accepted_attempts": sum(1 for row in gate_attempt_rows if row["shadow_gate_accepted"]),
        "shadow_gate_attempt_reasons": dict(Counter(row["shadow_gate_reason"] for row in gate_attempt_rows)),
        "shadow_gate_improvement_cp": {
            "mean": None if not gate_improvements else mean(gate_improvements),
            "median": None if not gate_improvements else median(gate_improvements),
            "min": None if not gate_improvements else min(gate_improvements),
            "max": None if not gate_improvements else max(gate_improvements),
        },
        "result_counts": dict(Counter(g["result_bucket"] for g in games_with_probe)),
        "termination_counts": dict(Counter(g["termination"] for g in games_with_probe)),
    }
    return {
        "combined": combined,
        "files": file_summaries,
        "games": all_games,
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    c = payload["combined"]
    lines = [
        "# V13.5 Shadow Value Audit",
        "",
        f"- files: {c['files']}",
        f"- games with probe: {c['games_with_probe']}/{c['games']}",
        f"- probes: {c['total_probes']}",
        f"- disagreements: {c['total_disagreements']} ({100.0 * c['disagreement_rate']:.1f}%)",
        f"- games with any disagreement: {c['games_with_disagreement']}",
        f"- games with early disagreement <= ply {c['early_ply_threshold']}: {c['games_with_early_disagreement']}",
        f"- shadow gate events: {c['shadow_gate_events']} in {c['shadow_gate_games']} games",
        f"- shadow gate attempts: {c.get('shadow_gate_attempts', 0)} "
        f"(accepted attempts: {c.get('shadow_gate_accepted_attempts', 0)})",
        "",
        "## Outcome Split",
        "",
        "| split | games | W-L-D | loss rate | mate/longcheck rate | avg disagreement rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("with disagreement", c["by_disagreement"]["with"]),
        ("without disagreement", c["by_disagreement"]["without"]),
        ("with early disagreement", c["by_early_disagreement"]["with"]),
        ("without early disagreement", c["by_early_disagreement"]["without"]),
        ("with shadow gate event", c["by_shadow_gate_event"]["with"]),
        ("without shadow gate event", c["by_shadow_gate_event"]["without"]),
    ):
        avg = row["avg_disagreement_rate"]
        avg_s = "n/a" if avg is None else f"{100.0 * avg:.1f}%"
        lines.append(
            f"| {label} | {row['games']} | {row['wins']}-{row['losses']}-{row['draws']} | "
            f"{100.0 * row['loss_rate']:.1f}% | {100.0 * row['mate_or_longcheck_rate']:.1f}% | {avg_s} |"
        )
    lines += [
        "",
        "## Worst Games By Disagreement Rate",
        "",
        "| source | game | result | term | side | opening | probes | disagreements | first ply |",
        "|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    games = sorted(
        [g for g in payload["games"] if g["probes"] > 0],
        key=lambda g: (-float(g["disagreement_rate"]), str(g["source_json"]), int(g["game_index"])),
    )
    for g in games[:20]:
        source = Path(str(g["source_json"])).name
        first = "" if g["first_disagreement_ply"] is None else str(g["first_disagreement_ply"])
        lines.append(
            f"| {source} | {g['game_index']} | {g['result']} | {g['termination']} | "
            f"{g['our_side']} | {g['opening_id']} | {g['probes']} | {g['disagreements']} | {first} |"
        )
    if c.get("shadow_gate_attempts", 0):
        lines += [
            "",
            "## Shadow Gate Attempt Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
        for reason, count in sorted(c.get("shadow_gate_attempt_reasons", {}).items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"| {reason or 'unknown'} | {count} |")
        imp = c.get("shadow_gate_improvement_cp", {})
        lines += [
            "",
            "## Shadow Gate Improvement CP",
            "",
            f"- mean: {imp.get('mean')}",
            f"- median: {imp.get('median')}",
            f"- min: {imp.get('min')}",
            f"- max: {imp.get('max')}",
        ]
    gated = [g for g in payload["games"] if g.get("has_shadow_gate_event")]
    if gated:
        lines += [
            "",
            "## Games With Shadow Gate Events",
            "",
            "| source | game | result | term | side | opening | events | first gate ply |",
            "|---|---:|---|---|---|---|---:|---:|",
        ]
        for g in sorted(gated, key=lambda row: (str(row["source_json"]), int(row["game_index"]))):
            source = Path(str(g["source_json"])).name
            first = "" if g["first_shadow_gate_ply"] is None else str(g["first_shadow_gate_ply"])
            lines.append(
                f"| {source} | {g['game_index']} | {g['result']} | {g['termination']} | "
                f"{g['our_side']} | {g['opening_id']} | {g['shadow_gate_events']} | {first} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_or_dir", nargs="+")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--early-ply-threshold", type=int, default=80)
    args = parser.parse_args()

    files = _input_files(list(args.json_or_dir))
    payload = _audit_files(files, int(args.early_ply_threshold))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    printable = {k: v for k, v in payload.items() if k != "games"}
    print(json.dumps(printable["combined"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
