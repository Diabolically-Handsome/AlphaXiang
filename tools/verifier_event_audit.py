#!/usr/bin/env python3
"""Audit root verifier events from external_arena JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _score_rate(our_wins: int, opp_wins: int, draws: int) -> float:
    total = our_wins + opp_wins + draws
    if total <= 0:
        return 0.0
    return (our_wins + 0.5 * draws) / total


def _collect_json_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("external_arena_*.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if resolved.is_file():
                paths.append(resolved)
                seen.add(resolved)
    return paths


def _candidate_by_move(event: dict[str, Any], move_uci: str) -> dict[str, Any]:
    for row in event.get("candidates", []) or []:
        if str(row.get("move_uci", "")) == move_uci:
            return row
    return {}


def _audit_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    our_wins = int(payload.get("our_wins", 0))
    opp_wins = int(payload.get("opp_wins", 0))
    draws = int(payload.get("draws", 0))
    total_games = int(payload.get("games", our_wins + opp_wins + draws))
    score = float(payload.get("score_rate", _score_rate(our_wins, opp_wins, draws)))
    per_game = payload.get("per_game", []) or []

    events: list[dict[str, Any]] = []
    for game_pos, game in enumerate(per_game, start=1):
        guard_events = game.get("guard_events", []) or []
        for event_pos, event in enumerate(guard_events, start=1):
            original = str(event.get("original_move_uci", ""))
            replacement = str(event.get("replacement_move_uci", ""))
            original_row = _candidate_by_move(event, original)
            replacement_row = _candidate_by_move(event, replacement)
            events.append(
                {
                    "game": game_pos,
                    "event": event_pos,
                    "ply": event.get("ply"),
                    "side": event.get("side", game.get("our_side", "")),
                    "result": game.get("result", ""),
                    "termination": game.get("termination", ""),
                    "guard_type": event.get("guard_type", ""),
                    "original_move_uci": original,
                    "replacement_move_uci": replacement,
                    "original_child_eval_cp_opponent_pov": event.get(
                        "original_child_eval_cp_opponent_pov"
                    ),
                    "replacement_child_eval_cp_opponent_pov": event.get(
                        "replacement_child_eval_cp_opponent_pov"
                    ),
                    "improvement_cp": event.get("improvement_cp"),
                    "margin_cp": event.get("margin_cp"),
                    "danger_threshold_cp": event.get("danger_threshold_cp"),
                    "original_mate_in": original_row.get("mate_in"),
                    "replacement_mate_in": replacement_row.get("mate_in"),
                    "verifier_mode": (
                        replacement_row.get("verifier_mode")
                        or original_row.get("verifier_mode")
                    ),
                    "fen_before": event.get("fen_before", ""),
                }
            )

    return {
        "path": str(path),
        "name": path.parent.name,
        "games": total_games,
        "our_wins": our_wins,
        "opp_wins": opp_wins,
        "draws": draws,
        "score_rate": score,
        "termination_counts": payload.get("termination_counts", {}) or {},
        "events": events,
        "events_per_game": (len(events) / total_games) if total_games else 0.0,
        "symbolic_guard_summary": payload.get("symbolic_guard_summary", {}) or {},
    }


def _combined_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    games = sum(int(r["games"]) for r in runs)
    our_wins = sum(int(r["our_wins"]) for r in runs)
    opp_wins = sum(int(r["opp_wins"]) for r in runs)
    draws = sum(int(r["draws"]) for r in runs)
    terminations: dict[str, int] = {}
    for run in runs:
        for key, value in run.get("termination_counts", {}).items():
            terminations[str(key)] = terminations.get(str(key), 0) + int(value)
    events = sum(len(r["events"]) for r in runs)
    return {
        "games": games,
        "our_wins": our_wins,
        "opp_wins": opp_wins,
        "draws": draws,
        "score_rate": _score_rate(our_wins, opp_wins, draws),
        "termination_counts": terminations,
        "events": events,
        "events_per_game": (events / games) if games else 0.0,
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _termination_text(counts: dict[str, Any]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _render_markdown(runs: list[dict[str, Any]], combined: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Verifier Event Audit")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(
        "| run | W-L-D | score | terminations | events | events/game |"
    )
    lines.append("|---|---:|---:|---|---:|---:|")
    for run in runs:
        wld = f"{run['our_wins']}-{run['opp_wins']}-{run['draws']}"
        lines.append(
            f"| `{run['name']}` | {wld} / {run['games']} | "
            f"{_format_pct(float(run['score_rate']))} | "
            f"{_termination_text(run['termination_counts'])} | "
            f"{len(run['events'])} | {float(run['events_per_game']):.3f} |"
        )
    lines.append("")
    lines.append(
        f"Combined: {combined['our_wins']}W-{combined['opp_wins']}L-"
        f"{combined['draws']}D / {combined['games']} = "
        f"{_format_pct(float(combined['score_rate']))}; "
        f"events={combined['events']} "
        f"({float(combined['events_per_game']):.3f}/game)."
    )
    lines.append("")
    lines.append("## Events")
    lines.append("")
    lines.append(
        "| run | game | ply | side | result | termination | type | move | cp opp POV | mate |"
    )
    lines.append("|---|---:|---:|---|---|---|---|---|---|---|")
    any_events = False
    for run in runs:
        for event in run["events"]:
            any_events = True
            move = (
                f"{event['original_move_uci']} -> "
                f"{event['replacement_move_uci']}"
            )
            cp = (
                f"{event['original_child_eval_cp_opponent_pov']} -> "
                f"{event['replacement_child_eval_cp_opponent_pov']} "
                f"(+{event['improvement_cp']})"
            )
            mate = (
                f"{event['original_mate_in']} -> "
                f"{event['replacement_mate_in']}"
            )
            lines.append(
                f"| `{run['name']}` | {event['game']} | {event['ply']} | "
                f"{event['side']} | {event['result']} | {event['termination']} | "
                f"{event['guard_type']} | `{move}` | {cp} | {mate} |"
            )
    if not any_events:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="external_arena JSON files or directories")
    parser.add_argument("--out-md", default="", help="Optional Markdown output path")
    parser.add_argument("--out-json", default="", help="Optional JSON output path")
    args = parser.parse_args()

    paths = _collect_json_paths(args.inputs)
    if not paths:
        raise SystemExit("no external_arena_*.json files found")

    runs = [_audit_one(path) for path in paths]
    combined = _combined_summary(runs)
    output = {"runs": runs, "combined": combined}

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = _render_markdown(runs, combined)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
