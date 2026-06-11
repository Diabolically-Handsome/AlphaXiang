"""Summarize AlphaXiang arena JSON files into a compact table.

The script is intentionally tolerant: missing globs are skipped so long-running
panel scripts can call it at the end without failing the whole run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        matches = [Path(p) for p in glob.glob(item)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(item)
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def _score_elo(score: float) -> float:
    if score <= 1e-4:
        return -2000.0
    if score >= 1.0 - 1e-4:
        return 2000.0
    return -400.0 * math.log10((1.0 - score) / score)


def _label_external(path: Path, payload: dict[str, Any]) -> str:
    cfg = payload.get("config") or {}
    opp_engine = cfg.get("opp_engine", "unknown")
    depth = payload.get("opp_depth")
    noise = float(cfg.get("opp_noise_ratio") or 0.0)
    value_source = cfg.get("our_value_source")
    parts = [str(opp_engine)]
    if depth:
        parts.append(f"d{depth}")
    if noise:
        parts.append(f"noise{noise:g}")
    if value_source and value_source != "scalar":
        parts.append(f"value={value_source}")
    return " ".join(parts) or path.parent.name


def _external_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wins = int(payload.get("our_wins", 0))
    losses = int(payload.get("opp_wins", 0))
    draws = int(payload.get("draws", 0))
    games = int(payload.get("games") or wins + losses + draws)
    score = float(payload.get("score_rate", (wins + 0.5 * draws) / max(games, 1)))
    guard_summary = payload.get("symbolic_guard_summary") or {}
    return {
        "label": _label_external(path, payload),
        "path": str(path),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": score,
        "elo_estimate": float(payload.get("elo_estimate", _score_elo(score))),
        "avg_plies": payload.get("avg_plies"),
        "guard_events": int(guard_summary.get("events") or 0),
        "guard_games": int(guard_summary.get("games_with_events") or 0),
        "kind": "external_arena",
    }


def _label_cnn(path: Path, payload: dict[str, Any]) -> str:
    value = payload.get("cnn_weights") or payload.get("cnn_engine") or "cnn"
    return f"cnn {Path(str(value)).name}"


def _cnn_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wins = int(payload.get("transformer_wins", 0))
    losses = int(payload.get("cnn_wins", 0))
    draws = int(payload.get("draws", 0))
    games = int(payload.get("games") or wins + losses + draws)
    score = float(payload.get("score_rate", (wins + 0.5 * draws) / max(games, 1)))
    return {
        "label": _label_cnn(path, payload),
        "path": str(path),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": score,
        "elo_estimate": _score_elo(score),
        "avg_plies": payload.get("avg_plies"),
        "guard_events": 0,
        "guard_games": 0,
        "kind": "transformer_vs_cnn",
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| cell | games | W-L-D | score | Elo vs opp | guard | source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        guard = "-"
        if int(row.get("guard_events") or 0) > 0:
            guard = f"{row['guard_events']}e/{row['guard_games']}g"
        lines.append(
            f"| {row['label']} | {row['games']} | "
            f"{row['wins']}-{row['losses']}-{row['draws']} | "
            f"{row['score_rate']:.3f} | {row['elo_estimate']:+.0f} | {guard} | "
            f"`{row['path']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-json", nargs="*", default=[])
    parser.add_argument("--cnn-json", nargs="*", default=[])
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in _expand(args.external_json):
        rows.append(_external_row(path))
    for path in _expand(args.cnn_json):
        rows.append(_cnn_row(path))
    rows.sort(key=lambda r: (r["label"], r["path"]))

    for row in rows:
        print(
            f"{row['label']}: {row['wins']}-{row['losses']}-{row['draws']} "
            f"score={row['score_rate']:.3f} elo={row['elo_estimate']:+.0f} "
            f"guard={row.get('guard_events', 0)}e/{row.get('guard_games', 0)}g "
            f"path={row['path']}",
            flush=True,
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_out:
        _write_markdown(rows, Path(args.markdown_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
