#!/usr/bin/env python3
"""Compare baseline shadow arena logs against shadow-gated arena logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.glob("external_arena_*.json")))
        elif path.is_file():
            out.append(path)
        else:
            raise FileNotFoundError(path)
    if not out:
        raise FileNotFoundError("no external_arena_*.json files found")
    return out


def _score(result: str) -> float:
    if result == "our_win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _bucket(result: str) -> str:
    if result == "our_win":
        return "win"
    if result == "opp_win":
        return "loss"
    return "draw"


def _gate_events(game: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in (game.get("guard_events", []) or [])
        if str(event.get("guard_type", "")) == "shadow_value_disagreement_verifier"
    ]


def _shadow_disagreements(game: dict[str, Any]) -> int:
    total = 0
    for row in game.get("search_stats", []) or []:
        probe = row.get("shadow_value_probe")
        if isinstance(probe, dict) and bool(probe.get("disagrees_with_actual")):
            total += 1
    return total


def _shadow_probes(game: dict[str, Any]) -> int:
    return sum(
        1 for row in (game.get("search_stats", []) or [])
        if isinstance(row.get("shadow_value_probe"), dict)
    )


def _game_key(game: dict[str, Any]) -> str:
    opening = game.get("opening_id")
    opening_index = game.get("opening_index")
    if opening:
        return f"opening_id:{opening}:game_index:{game.get('index')}"
    if opening_index is not None:
        return f"opening_index:{opening_index}:game_index:{game.get('index')}"
    return f"game_index:{game.get('index')}"


def _load_records(paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = _files(paths)
    records: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_rows.append({
            "path": str(path),
            "score_rate": payload.get("score_rate"),
            "our_wins": payload.get("our_wins"),
            "opp_wins": payload.get("opp_wins"),
            "draws": payload.get("draws"),
            "termination_counts": payload.get("termination_counts", {}),
            "shadow_value_summary": payload.get("shadow_value_summary", {}),
            "symbolic_guard_summary": payload.get("symbolic_guard_summary", {}),
        })
        for game in payload.get("per_game", []) or []:
            gates = _gate_events(game)
            records.append({
                "source_json": str(path),
                "key": _game_key(game),
                "index": int(game.get("index", -1)),
                "our_side": str(game.get("our_side", "")),
                "opening_id": str(game.get("opening_id", "")),
                "opening_index": game.get("opening_index"),
                "result": str(game.get("result", "")),
                "bucket": _bucket(str(game.get("result", ""))),
                "score": _score(str(game.get("result", ""))),
                "termination": str(game.get("termination", "")),
                "plies": int(game.get("plies", 0) or 0),
                "shadow_probes": _shadow_probes(game),
                "shadow_disagreements": _shadow_disagreements(game),
                "shadow_gate_events": len(gates),
                "shadow_gate_replacements": [
                    {
                        "ply": event.get("ply"),
                        "original": event.get("original_move_uci"),
                        "replacement": event.get("replacement_move_uci"),
                        "improvement_cp": event.get("improvement_cp"),
                    }
                    for event in gates
                ],
            })
    return file_rows, records


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    gates = [row for row in records if int(row["shadow_gate_events"]) > 0]
    return {
        "games": n,
        "wins": sum(1 for row in records if row["bucket"] == "win"),
        "losses": sum(1 for row in records if row["bucket"] == "loss"),
        "draws": sum(1 for row in records if row["bucket"] == "draw"),
        "score_rate": 0.0 if n == 0 else mean([float(row["score"]) for row in records]),
        "mate_losses": sum(1 for row in records if row["bucket"] == "loss" and row["termination"] == "mate"),
        "longcheck_losses": sum(1 for row in records if row["bucket"] == "loss" and row["termination"] == "longcheck"),
        "termination_counts": dict(Counter(row["termination"] for row in records)),
        "shadow_probes": sum(int(row["shadow_probes"]) for row in records),
        "shadow_disagreements": sum(int(row["shadow_disagreements"]) for row in records),
        "shadow_gate_events": sum(int(row["shadow_gate_events"]) for row in records),
        "games_with_shadow_gate": len(gates),
    }


def _paired_rows(base: list[dict[str, Any]], gated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_base = {row["key"]: row for row in base}
    out: list[dict[str, Any]] = []
    for gated_row in gated:
        base_row = by_base.get(gated_row["key"])
        if base_row is None:
            continue
        out.append({
            "key": gated_row["key"],
            "opening_id": gated_row["opening_id"] or base_row["opening_id"],
            "opening_index": gated_row["opening_index"],
            "baseline_result": base_row["result"],
            "gated_result": gated_row["result"],
            "baseline_termination": base_row["termination"],
            "gated_termination": gated_row["termination"],
            "score_delta": float(gated_row["score"]) - float(base_row["score"]),
            "gated_shadow_gate_events": gated_row["shadow_gate_events"],
            "gated_replacements": gated_row["shadow_gate_replacements"],
        })
    return out


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    b = payload["baseline_summary"]
    g = payload["gated_summary"]
    p = payload["paired_summary"]
    lines = [
        "# V13.5 Shadow Gate Compare",
        "",
        "| run | games | W-L-D | score | mate losses | longcheck losses | shadow gate events |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| baseline | {b['games']} | {b['wins']}-{b['losses']}-{b['draws']} | {100*b['score_rate']:.1f}% | {b['mate_losses']} | {b['longcheck_losses']} | {b['shadow_gate_events']} |",
        f"| gated | {g['games']} | {g['wins']}-{g['losses']}-{g['draws']} | {100*g['score_rate']:.1f}% | {g['mate_losses']} | {g['longcheck_losses']} | {g['shadow_gate_events']} |",
        "",
        "## Paired Delta",
        "",
        f"- paired games: {p['paired_games']}",
        f"- score delta: {p['score_delta']:+.3f}",
        f"- improved games: {p['improved_games']}",
        f"- worsened games: {p['worsened_games']}",
        f"- unchanged games: {p['unchanged_games']}",
        "",
        "## Gated Replacements",
        "",
        "| key | base | gated | delta | gated term | events |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in payload["paired_games"]:
        if int(row["gated_shadow_gate_events"]) <= 0:
            continue
        lines.append(
            f"| {row['key']} | {row['baseline_result']} | {row['gated_result']} | "
            f"{row['score_delta']:+.1f} | {row['gated_termination']} | {row['gated_shadow_gate_events']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--gated", nargs="+", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    baseline_files, baseline_records = _load_records(list(args.baseline))
    gated_files, gated_records = _load_records(list(args.gated))
    paired = _paired_rows(baseline_records, gated_records)
    paired_summary = {
        "paired_games": len(paired),
        "score_delta": sum(float(row["score_delta"]) for row in paired),
        "improved_games": sum(1 for row in paired if float(row["score_delta"]) > 0),
        "worsened_games": sum(1 for row in paired if float(row["score_delta"]) < 0),
        "unchanged_games": sum(1 for row in paired if float(row["score_delta"]) == 0),
    }
    payload = {
        "baseline_files": baseline_files,
        "gated_files": gated_files,
        "baseline_summary": _summary(baseline_records),
        "gated_summary": _summary(gated_records),
        "paired_summary": paired_summary,
        "paired_games": paired,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    print(json.dumps({k: payload[k] for k in ("baseline_summary", "gated_summary", "paired_summary")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
