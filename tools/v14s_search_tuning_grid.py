#!/usr/bin/env python3
"""V14S search-only tuning runner for the fixed V13 release checkpoint.

This deliberately does not train or edit model weights.  It wraps
``tools/external_arena.py`` with the V13.3 inference safety stack and records a
small, comparable grid of MCTS search knobs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_CKPT = (
    "/home/laure/alphaxiang/training_runs/"
    "run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/"
    "snapshots/latest_step19000.pt"
)


def _parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _latest_arena_json(out_dir: Path) -> Path | None:
    files = sorted(out_dir.glob("external_arena_*.json"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _score_stderr(score: float, games: int) -> float:
    # Treat a game score in [0,1] as a bounded Bernoulli-like estimate.  This is
    # only for coarse smoke ranking; final promotion still needs much larger N.
    if games <= 0:
        return 0.0
    return (max(score * (1.0 - score), 0.0) / float(games)) ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    parser.add_argument("--output-dir", default="/home/laure/alphaxiang/v14s_search_tuning/coarse_d5_black")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026051401)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--opening-suite-path", default="")
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--max-openings", type=int, default=0)
    parser.add_argument("--parallel-games", type=int, default=1)
    parser.add_argument("--our-side", choices=["alternate", "red", "black"], default="black")
    parser.add_argument("--opp-depth", type=int, default=5)
    parser.add_argument("--opp-threads", type=int, default=1)
    parser.add_argument("--opp-hash-mb", type=int, default=128)
    parser.add_argument("--sims", default="8000")
    parser.add_argument("--c-puct", default="0.9,1.1,1.25,1.45")
    parser.add_argument("--c-puct-base", default="1.0")
    parser.add_argument("--c-puct-factor", default="0.0")
    parser.add_argument("--q-weight", default="0.85,1.0,1.15")
    parser.add_argument("--q-clip", default="1.0")
    parser.add_argument("--fpu-reduction-root", default="-1.0")
    parser.add_argument("--fpu-reduction-tree", default="-1.0")
    parser.add_argument("--temperature-move", default="0.02")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--repeat-limit", type=int, default=6)
    parser.add_argument("--repeat-min-ply", type=int, default=30)
    parser.add_argument("--no-capture-limit", type=int, default=60)
    parser.add_argument("--include-baseline-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ship-safety", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")

    sims_values = _parse_csv_ints(args.sims)
    cpuct_values = _parse_csv_floats(args.c_puct)
    cpuct_base_values = _parse_csv_floats(args.c_puct_base)
    cpuct_factor_values = _parse_csv_floats(args.c_puct_factor)
    q_weight_values = _parse_csv_floats(args.q_weight)
    q_clip_values = _parse_csv_floats(args.q_clip)
    fpu_root_values = _parse_csv_floats(args.fpu_reduction_root)
    fpu_tree_values = _parse_csv_floats(args.fpu_reduction_tree)
    temp_values = _parse_csv_floats(args.temperature_move)
    combos = list(itertools.product(
        sims_values,
        cpuct_values,
        cpuct_base_values,
        cpuct_factor_values,
        q_weight_values,
        q_clip_values,
        fpu_root_values,
        fpu_tree_values,
        temp_values,
    ))
    planned_games_per_combo = (
        int(args.max_openings) * int(args.games_per_opening)
        if str(args.opening_suite_path) and int(args.max_openings) > 0
        else int(args.games)
    )

    baseline = (8000, 1.25, 1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.02)
    if args.include_baseline_first and baseline in combos:
        combos.remove(baseline)
        combos.insert(0, baseline)

    out_root = Path(args.output_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "v14s_search_grid_summary.json"
    summary_md_path = out_root / "v14s_search_grid_summary.md"
    rows: list[dict[str, Any]] = []

    print(
        f"V14S search grid: {len(combos)} combo(s), games={args.games}, "
        f"planned_games_per_combo={planned_games_per_combo}, "
        f"side={args.our_side}, depth={args.opp_depth}",
        flush=True,
    )

    for combo_idx, (
        sims,
        c_puct,
        c_puct_base,
        c_puct_factor,
        q_weight,
        q_clip,
        fpu_root,
        fpu_tree,
        temperature,
    ) in enumerate(combos, start=1):
        tag = (
            f"sims{sims}_cpuct{c_puct:g}_base{c_puct_base:g}_fac{c_puct_factor:g}_"
            f"q{q_weight:g}_clip{q_clip:g}_fpuR{fpu_root:g}_fpuT{fpu_tree:g}_"
            f"temp{temperature:g}_{args.our_side}"
        )
        combo_dir = out_root / tag
        cmd = [
            str(args.python),
            str(_REPO / "tools" / "external_arena.py"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(combo_dir),
            "--games", str(int(args.games)),
            "--our-side", str(args.our_side),
            "--games-per-opening", str(int(args.games_per_opening)),
            "--max-openings", str(int(args.max_openings)),
            "--parallel-games", str(int(args.parallel_games)),
            "--cross-game-batch-cap", "96",
            "--device", str(args.device),
            "--seed", str(int(args.seed) + combo_idx * 1009),
            "--opp-engine", "pikafish",
            "--opp-depth", str(int(args.opp_depth)),
            "--opp-threads", str(int(args.opp_threads)),
            "--opp-hash-mb", str(int(args.opp_hash_mb)),
            "--our-sims", str(int(sims)),
            "--our-c-puct", str(float(c_puct)),
            "--our-c-puct-base", str(float(c_puct_base)),
            "--our-c-puct-factor", str(float(c_puct_factor)),
            "--our-q-weight", str(float(q_weight)),
            "--our-q-clip", str(float(q_clip)),
            "--our-fpu-reduction-root", str(float(fpu_root)),
            "--our-fpu-reduction-tree", str(float(fpu_tree)),
            "--our-temperature-move", str(float(temperature)),
            "--max-plies", str(int(args.max_plies)),
            "--repeat-limit", str(int(args.repeat_limit)),
            "--repeat-min-ply", str(int(args.repeat_min_ply)),
            "--no-capture-limit", str(int(args.no_capture_limit)),
        ]
        if str(args.opening_suite_path):
            cmd += ["--opening-suite-path", str(args.opening_suite_path)]
        if args.enable_ship_safety:
            cmd += [
                "--our-root-mate1-blunder-guard",
                "--our-tactical-mate1-extension",
                "--our-tactical-mate2-extension",
            ]

        print(f"[{combo_idx}/{len(combos)}] {tag}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        started = time.monotonic()
        rc = subprocess.run(cmd).returncode
        seconds = time.monotonic() - started
        arena_json = _latest_arena_json(combo_dir)
        payload = _load_json(arena_json)
        row: dict[str, Any] = {
            "tag": tag,
            "returncode": int(rc),
            "seconds": float(seconds),
            "sims": int(sims),
            "c_puct": float(c_puct),
            "c_puct_base": float(c_puct_base),
            "c_puct_factor": float(c_puct_factor),
            "q_weight": float(q_weight),
            "q_clip": float(q_clip),
            "fpu_reduction_root": float(fpu_root),
            "fpu_reduction_tree": float(fpu_tree),
            "temperature_move": float(temperature),
            "our_side": str(args.our_side),
            "opp_depth": int(args.opp_depth),
            "opening_suite_path": str(args.opening_suite_path),
            "games_per_opening": int(args.games_per_opening),
            "max_openings": int(args.max_openings),
            "json": str(arena_json) if arena_json is not None else None,
        }
        if payload is not None:
            score = float(payload.get("score_rate", 0.0))
            games = int(payload.get("games", args.games))
            row.update(
                {
                    "games": games,
                    "our_wins": int(payload.get("our_wins", 0)),
                    "opp_wins": int(payload.get("opp_wins", 0)),
                    "draws": int(payload.get("draws", 0)),
                    "score_rate": score,
                    "score_stderr": _score_stderr(score, games),
                    "elo_estimate": payload.get("elo_estimate"),
                    "avg_plies": payload.get("avg_plies"),
                    "termination_counts": payload.get("termination_counts"),
                    "symbolic_guard_summary": payload.get("symbolic_guard_summary"),
                }
            )
        rows.append(row)
        rows_sorted = sorted(rows, key=lambda item: float(item.get("score_rate", -1.0)), reverse=True)
        summary_path.write_text(json.dumps(rows_sorted, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = [
            "# V14S Search Grid Summary",
            "",
            f"- checkpoint: `{checkpoint}`",
            f"- side: `{args.our_side}`",
            f"- opponent: Pikafish d{args.opp_depth}",
            f"- games argument: `{args.games}`",
            f"- planned games per combo: `{planned_games_per_combo}`",
            f"- opening suite: `{args.opening_suite_path or 'standard startpos'}`",
            f"- games per opening: `{args.games_per_opening if args.opening_suite_path else 'n/a'}`",
            f"- max openings: `{args.max_openings if args.opening_suite_path else 'n/a'}`",
            "",
            "| rank | tag | W-L-D | score | stderr | avg plies | json |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
        for rank, item in enumerate(rows_sorted, start=1):
            wld = f"{item.get('our_wins', '?')}-{item.get('opp_wins', '?')}-{item.get('draws', '?')}"
            lines.append(
                f"| {rank} | `{item['tag']}` | {wld} | "
                f"{float(item.get('score_rate', 0.0)):.3f} | "
                f"{float(item.get('score_stderr', 0.0)):.3f} | "
                f"{float(item.get('avg_plies', 0.0)):.1f} | `{item.get('json')}` |"
            )
        summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"  rc={rc} score={row.get('score_rate')} W-L-D="
            f"{row.get('our_wins')}-{row.get('opp_wins')}-{row.get('draws')} "
            f"dt={seconds:.0f}s",
            flush=True,
        )

    if args.dry_run:
        print("dry-run complete", flush=True)
    else:
        print(f"DONE: {summary_path}", flush=True)
        print(f"DONE: {summary_md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
