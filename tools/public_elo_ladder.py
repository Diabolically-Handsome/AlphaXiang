"""Run a reproducible Pikafish UCI_Elo ladder for public Elo reference.

This is not an official human rating.  It anchors AlphaXiang against Pikafish's
documented UCI_LimitStrength/UCI_Elo levels, then converts each match score into
an Elo estimate relative to that public engine setting.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ARENA = REPO_ROOT / "tools" / "external_arena.py"
PIKAFISH_UCI_DOC = "https://www.pikafish.com/wiki/index.php?title=UCI%E9%80%89%E9%A1%B9"


def _parse_levels(text: str) -> list[int]:
    levels: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if not 1280 <= value <= 3133:
            raise argparse.ArgumentTypeError(f"Pikafish UCI_Elo out of documented range: {value}")
        levels.append(value)
    if not levels:
        raise argparse.ArgumentTypeError("at least one level is required")
    return levels


def _elo_delta(score: float) -> float:
    score = max(1e-4, min(1.0 - 1e-4, float(score)))
    return -400.0 * math.log10((1.0 - score) / score)


def _latest_json(path: Path) -> Path | None:
    files = sorted(path.glob("external_arena_*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _uci_option_names(binary: Path) -> set[str]:
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(binary.parent),
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    names: set[str] = set()
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            if line.startswith("option name "):
                rest = line[len("option name "):]
                name = rest.split(" type ", 1)[0].strip()
                if name:
                    names.add(name)
            if "uciok" in line:
                break
    finally:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()
    return names


def _validate_pikafish_uci_elo(binary: Path) -> None:
    names = _uci_option_names(binary)
    required = {"UCI_LimitStrength", "UCI_Elo"}
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(
            "The selected Pikafish binary does not expose the UCI Elo limiting "
            f"options {missing}. Refusing to run a misleading public-Elo ladder. "
            f"binary={binary}; options={sorted(names)}"
        )


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_level(args: argparse.Namespace, level: int, index: int) -> dict[str, Any]:
    out_dir = Path(args.output_root) / f"pika_uci_elo_{level}_g{args.games_per_level}_sims{args.our_sims}"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _latest_json(out_dir)
    if existing is not None and not args.force:
        payload = _load_payload(existing)
        payload["_json_path"] = str(existing)
        payload["_skipped_existing"] = True
        return payload

    cmd = [
        sys.executable,
        str(EXTERNAL_ARENA),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--output-dir",
        str(out_dir),
        "--games",
        str(args.games_per_level),
        "--our-side",
        args.our_side,
        "--parallel-games",
        str(args.parallel_games),
        "--seed",
        str(int(args.seed) + index * 1009 + level),
        "--device",
        args.device,
        "--opp-engine",
        "pikafish",
        "--pikafish-binary",
        str(Path(args.pikafish_binary).resolve()),
        "--opp-movetime-ms",
        str(args.opp_movetime_ms),
        "--opp-uci-elo",
        str(level),
        "--opp-threads",
        str(args.opp_threads),
        "--opp-hash-mb",
        str(args.opp_hash_mb),
        "--our-sims",
        str(args.our_sims),
        "--our-q-weight",
        str(args.our_q_weight),
        "--our-temperature-move",
        str(args.our_temperature_move),
        "--max-plies",
        str(args.max_plies),
    ]
    if args.root_mate1_guard:
        cmd.append("--our-root-mate1-blunder-guard")
    if args.root_mate2_guard:
        cmd.append("--our-root-mate2-blunder-guard")
    if args.tactical_mate1_extension:
        cmd.append("--our-tactical-mate1-extension")
    if args.tactical_mate2_extension:
        cmd.append("--our-tactical-mate2-extension")
    if int(args.root_forcing_check_guard_plies) > 0:
        cmd.extend([
            "--our-root-forcing-check-guard-plies",
            str(args.root_forcing_check_guard_plies),
        ])

    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    produced = _latest_json(out_dir)
    if produced is None:
        raise RuntimeError(f"arena finished but no JSON was written in {out_dir}")
    payload = _load_payload(produced)
    payload["_json_path"] = str(produced)
    payload["_skipped_existing"] = False
    return payload


def _summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    estimates: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for row in rows:
        cfg = row.get("config", {})
        level = int(row.get("opp_uci_elo") or cfg.get("opp_uci_elo") or 0)
        wins = int(row.get("our_wins", 0))
        losses = int(row.get("opp_wins", 0))
        draws = int(row.get("draws", 0))
        games = max(1, wins + losses + draws)
        score = float(row.get("score_rate", (wins + 0.5 * draws) / games))
        delta = float(row.get("elo_estimate", _elo_delta(score)))
        estimate = float(level + delta)
        # Scores near 50% carry more information for rating placement.
        weight = max(0.02, score * (1.0 - score)) * games
        weighted_num += estimate * weight
        weighted_den += weight
        if 0.10 <= score <= 0.90:
            estimates.append(estimate)
        table.append({
            "pika_uci_elo": level,
            "games": games,
            "wld": f"{wins}-{losses}-{draws}",
            "score": score,
            "elo_delta_vs_level": delta,
            "public_elo_estimate": estimate,
            "json": row.get("_json_path"),
            "skipped_existing": bool(row.get("_skipped_existing", False)),
        })

    center = median(estimates) if estimates else (weighted_num / weighted_den if weighted_den else None)
    weighted = weighted_num / weighted_den if weighted_den else None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "method": "Pikafish UCI_LimitStrength=true + UCI_Elo ladder",
        "source": PIKAFISH_UCI_DOC,
        "caveat": (
            "This is a public engine-strength reference, not an official human federation Elo. "
            "Pikafish documents UCI_Elo as calibrated to a Xiangqi engine league ladder."
        ),
        "estimated_public_elo_center": center,
        "estimated_public_elo_weighted": weighted,
        "rows": table,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }


def _write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "public_elo_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# AlphaXiang Public Elo Ladder",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Method: {summary['method']}",
        f"- Source: {summary['source']}",
        f"- Caveat: {summary['caveat']}",
        f"- Center estimate: `{summary['estimated_public_elo_center']}`",
        f"- Weighted estimate: `{summary['estimated_public_elo_weighted']}`",
        "",
        "| Pika UCI_Elo | W-L-D | Score | Delta | Estimate | JSON |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        md_lines.append(
            f"| {row['pika_uci_elo']} | {row['wld']} | {row['score']:.3f} | "
            f"{row['elo_delta_vs_level']:+.0f} | {row['public_elo_estimate']:.0f} | "
            f"`{row['json']}` |"
        )
    md_path = output_root / "public_elo_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", default="/home/laure/alphaxiang/public_elo_ladder")
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--levels", type=_parse_levels, default=_parse_levels("2200,2400,2600,2800"))
    p.add_argument("--games-per-level", type=int, default=10)
    p.add_argument("--our-side", choices=["alternate", "red", "black"], default="alternate")
    p.add_argument("--parallel-games", type=int, default=8)
    p.add_argument("--seed", type=int, default=202605110)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--opp-movetime-ms", type=int, default=200)
    p.add_argument("--opp-threads", type=int, default=1)
    p.add_argument("--opp-hash-mb", type=int, default=128)
    p.add_argument("--our-sims", type=int, default=8000)
    p.add_argument("--our-q-weight", type=float, default=1.0)
    p.add_argument("--our-temperature-move", type=float, default=0.02)
    p.add_argument("--max-plies", type=int, default=180)
    p.add_argument("--root-mate1-guard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--root-mate2-guard", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--root-forcing-check-guard-plies", type=int, default=0)
    p.add_argument("--tactical-mate1-extension", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tactical-mate2-extension", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="rerun levels even if a JSON already exists")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _validate_pikafish_uci_elo(Path(args.pikafish_binary).resolve())
    rows = []
    for i, level in enumerate(args.levels):
        rows.append(_run_level(args, level, i))
    summary = _summarize(args, rows)
    _write_summary(Path(args.output_root), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
