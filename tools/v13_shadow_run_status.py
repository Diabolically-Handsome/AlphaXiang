#!/usr/bin/env python3
"""Summarize V13.5 shadow/gated arena run status from logs and output JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


GAME_RE = re.compile(
    r"game\s+(?P<num>\d+)/(?P<total>\d+)\s+\(gi=(?P<gi>\d+)\)\s+"
    r"our_side=(?P<side>\w+)\s+result=(?P<result>\w+)\s+"
    r"plies=(?P<plies>\d+)\s+term=(?P<term>\w+)"
)
DONE_RE = re.compile(
    r"DONE:\s+(?P<wins>\d+)W\s+-\s+(?P<losses>\d+)L\s+-\s+"
    r"(?P<draws>\d+)D\s+over\s+(?P<games>\d+)\s+games"
)


def _latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime, default=None)


def _default_log(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return root / names[0]


def _default_log_glob(root: Path, *names: str, patterns: str | tuple[str, ...] = ()) -> Path:
    path = _default_log(root, *names)
    if path.is_file():
        return path
    pats = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    matches: list[Path] = []
    for pattern in pats:
        matches.extend(root.glob(pattern))
    latest = _latest([p for p in matches if p.is_file()])
    return latest if latest is not None else path


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score(result: str) -> float:
    if result == "our_win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _games_from_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GAME_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "num": int(m.group("num")),
                "total": int(m.group("total")),
                "gi": int(m.group("gi")),
                "our_side": m.group("side"),
                "result": m.group("result"),
                "plies": int(m.group("plies")),
                "termination": m.group("term"),
            }
        )
    return rows


def _done_from_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = DONE_RE.search(line)
        if not m:
            continue
        rows.append({k: int(v) for k, v in m.groupdict().items()})
    return rows


def _summary_from_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for row in games if row.get("result") == "our_win")
    losses = sum(1 for row in games if row.get("result") == "opp_win")
    draws = sum(1 for row in games if row.get("result") == "draw")
    n = len(games)
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": None if n == 0 else (wins + 0.5 * draws) / n,
        "mate_losses": sum(1 for row in games if row.get("result") == "opp_win" and row.get("termination") == "mate"),
        "longcheck_losses": sum(1 for row in games if row.get("result") == "opp_win" and row.get("termination") == "longcheck"),
        "termination_counts": dict(Counter(str(row.get("termination", "")) for row in games)),
    }


def _shadow_gate_events(game: dict[str, Any]) -> int:
    events = 0
    for event in game.get("guard_events", []) or []:
        if str(event.get("guard_type", "")) == "shadow_value_disagreement_verifier":
            events += 1
    return events


def _summary_from_arena_json(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    per_game = payload.get("per_game", []) or []
    games = [
        {
            "result": str(row.get("result", "")),
            "termination": str(row.get("termination", "")),
            "plies": int(row.get("plies", 0) or 0),
        }
        for row in per_game
    ]
    summary = _summary_from_games(games)
    summary.update(
        {
            "json_games": len(per_game),
            "reported_score_rate": payload.get("score_rate"),
            "shadow_value_summary": payload.get("shadow_value_summary", {}),
            "symbolic_guard_summary": payload.get("symbolic_guard_summary", {}),
            "shadow_gate_events": sum(_shadow_gate_events(row) for row in per_game),
            "source_json": "",
        }
    )
    return summary


def _phase(root: Path, log_games: list[dict[str, Any]], gated_log_games: list[dict[str, Any]] | None = None) -> str:
    decision = root / "d5_shadow_gate_decision.json"
    if decision.exists():
        return "d5 decision ready"
    compare = root / "d5_shadow_gate_compare.json"
    if compare.exists():
        return "d5 compare ready"
    if (root / "d5_shadow_gated").is_dir():
        gated_json = _latest(sorted((root / "d5_shadow_gated").glob("external_arena_*.json")))
        if gated_json is not None:
            return "d5 gated complete"
        if gated_log_games:
            latest = gated_log_games[-1]
            return f"d5 gated running ({latest['num']}/{latest['total']})"
        return "d5 gated running or pending"
    baseline_json = _latest(sorted((root / "d5_shadow").glob("external_arena_*.json"))) if (root / "d5_shadow").is_dir() else None
    if baseline_json is not None:
        return "d5 baseline complete or auditing"
    if log_games:
        latest = log_games[-1]
        return f"d5 baseline running ({latest['num']}/{latest['total']})"
    return "starting or no progress yet"


def build_status(root: Path, log: Path, gated_log: Path | None = None) -> dict[str, Any]:
    log_games = _games_from_log(log)
    done_rows = _done_from_log(log)
    gated_log_games = _games_from_log(gated_log) if gated_log is not None else []
    gated_done_rows = _done_from_log(gated_log) if gated_log is not None else []
    baseline_json_path = _latest(sorted((root / "d5_shadow").glob("external_arena_*.json"))) if (root / "d5_shadow").is_dir() else None
    gated_json_path = _latest(sorted((root / "d5_shadow_gated").glob("external_arena_*.json"))) if (root / "d5_shadow_gated").is_dir() else None
    baseline = _summary_from_arena_json(_load_json(baseline_json_path))
    gated = _summary_from_arena_json(_load_json(gated_json_path))
    if baseline is not None and baseline_json_path is not None:
        baseline["source_json"] = str(baseline_json_path)
    if gated is not None and gated_json_path is not None:
        gated["source_json"] = str(gated_json_path)
    compare = _load_json(root / "d5_shadow_gate_compare.json")
    decision = _load_json(root / "d5_shadow_gate_decision.json")
    return {
        "root": str(root),
        "log": str(log),
        "phase": _phase(root, log_games, gated_log_games),
        "log_progress": {
            "games": log_games,
            "summary": _summary_from_games(log_games),
            "done_rows": done_rows,
            "latest_game": log_games[-1] if log_games else None,
        },
        "gated_log": "" if gated_log is None else str(gated_log),
        "gated_log_progress": {
            "games": gated_log_games,
            "summary": _summary_from_games(gated_log_games),
            "done_rows": gated_done_rows,
            "latest_game": gated_log_games[-1] if gated_log_games else None,
        },
        "baseline": baseline,
        "gated": gated,
        "compare": compare,
        "decision": decision,
    }


def _fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    progress = payload["log_progress"]
    summary = progress["summary"]
    latest = progress["latest_game"]
    lines = [
        "# V13.5 Shadow WDL Run Status",
        "",
        f"- phase: {payload['phase']}",
        f"- log games: {summary['games']} ({summary['wins']}-{summary['losses']}-{summary['draws']}, score {_fmt_score(summary['score_rate'])})",
        f"- log mate/longcheck losses: {summary['mate_losses']} / {summary['longcheck_losses']}",
    ]
    if latest:
        lines.append(
            f"- latest game: {latest['num']}/{latest['total']} {latest['result']} "
            f"{latest['plies']} plies {latest['termination']}"
        )
    gated_progress = payload.get("gated_log_progress", {})
    gated_summary = gated_progress.get("summary", {})
    gated_latest = gated_progress.get("latest_game")
    if gated_summary.get("games", 0):
        lines += [
            f"- gated log games: {gated_summary['games']} ({gated_summary['wins']}-{gated_summary['losses']}-{gated_summary['draws']}, score {_fmt_score(gated_summary['score_rate'])})",
            f"- gated log mate/longcheck losses: {gated_summary['mate_losses']} / {gated_summary['longcheck_losses']}",
        ]
        if gated_latest:
            lines.append(
                f"- latest gated game: {gated_latest['num']}/{gated_latest['total']} {gated_latest['result']} "
                f"{gated_latest['plies']} plies {gated_latest['termination']}"
            )
    for label in ("baseline", "gated"):
        row = payload.get(label)
        if not row:
            continue
        lines += [
            "",
            f"## {label.title()}",
            "",
            f"- games: {row['games']} ({row['wins']}-{row['losses']}-{row['draws']}, score {_fmt_score(row['score_rate'])})",
            f"- mate/longcheck losses: {row['mate_losses']} / {row['longcheck_losses']}",
            f"- shadow gate events: {row.get('shadow_gate_events', 0)}",
            f"- source: {row.get('source_json', '')}",
        ]
    decision = payload.get("decision")
    if decision:
        observed = decision.get("observed", {})
        lines += [
            "",
            "## Decision",
            "",
            f"- status: {decision.get('status')}",
            f"- passed: {decision.get('passed')}",
            f"- paired games: {observed.get('paired_games')}",
            f"- score delta: {observed.get('score_delta')}",
            f"- mate losses: {observed.get('baseline_mate_losses')} -> {observed.get('gated_mate_losses')}",
            f"- longcheck losses: {observed.get('baseline_longcheck_losses')} -> {observed.get('gated_longcheck_losses')}",
            f"- shadow gate events: {observed.get('shadow_gate_events')}",
        ]
        reasons = decision.get("reasons") or []
        if reasons:
            lines += ["", "### Reasons", ""]
            lines.extend(f"- {reason}" for reason in reasons)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/laure/alphaxiang/v13_shadow_wdl_probe")
    parser.add_argument("--log", default="")
    parser.add_argument("--gated-log", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    root = Path(args.root)
    log = Path(args.log) if args.log else _default_log(
        root,
        "v135_shadow_wdl_d5_full.log",
        "v135_shadow_wdl_d5_full_fixed.log",
    )
    gated_log = Path(args.gated_log) if args.gated_log else _default_log_glob(
        root,
        "v135_shadow_wdl_d5_gated.log",
        "v135_shadow_wdl_d5_gated_fixed.log",
        "v135_shadow_wdl_d5_gated_margin600.log",
        patterns=("v135_shadow_wdl_d5_gated*.log",),
    )
    payload = build_status(root, log, gated_log if gated_log.is_file() else None)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        _write_md(Path(args.out_md), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
