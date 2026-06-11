"""Run a v12.5 no-training MCTS diagnostic grid via external_arena.py.

This answers the first falsification question before spending GPU days on v13:
if the same checkpoint improves materially just by changing sims/c_puct/q_weight,
the bottleneck is search calibration rather than model capacity.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_root_noise_modes(text: str) -> list[bool]:
    out: list[bool] = []
    for item in text.split(","):
        t = item.strip().lower()
        if not t:
            continue
        if t in {"on", "true", "1", "yes"}:
            out.append(True)
        elif t in {"off", "false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"bad root-noise mode: {item!r}")
    return out


def _load_latest_arena_json(out_dir: Path) -> dict | None:
    files = sorted(out_dir.glob("external_arena_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="/home/laure/alphaxiang/v12_5_mcts_grid")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--parallel-games", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--opp-engine", default="pikafish",
                   choices=["pikafish", "fairy_sf", "elephantart", "eleeye"])
    p.add_argument("--opp-depth", type=int, default=3)
    p.add_argument("--opp-movetime-ms", type=int, default=0)
    p.add_argument("--opp-nodes", type=int, default=0)
    p.add_argument("--pikafish-binary", default="/home/laure/pikafish/pikafish")
    p.add_argument("--sims", default="800,1600,3200")
    p.add_argument("--c-puct", default="1.0,1.25,1.6")
    p.add_argument("--q-weight", default="0.75,1.0,1.25")
    p.add_argument("--q-clip", default="1.0")
    p.add_argument("--root-noise", default="off",
                   help="Comma-separated off/on. Default off; use off,on for exploration check.")
    p.add_argument("--temperature-move", type=float, default=0.1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    sims_values = _parse_csv_ints(args.sims)
    c_puct_values = _parse_csv_floats(args.c_puct)
    q_weight_values = _parse_csv_floats(args.q_weight)
    q_clip_values = _parse_csv_floats(args.q_clip)
    root_noise_values = _parse_root_noise_modes(args.root_noise)

    combos = list(itertools.product(
        sims_values, c_puct_values, q_weight_values, q_clip_values, root_noise_values,
    ))
    print(f"mcts grid: {len(combos)} combo(s), games={args.games} each", flush=True)

    summary: list[dict] = []
    for idx, (sims, c_puct, q_weight, q_clip, root_noise) in enumerate(combos, start=1):
        tag = (
            f"sims{sims}_cpuct{c_puct:g}_q{q_weight:g}_clip{q_clip:g}_"
            f"noise{'on' if root_noise else 'off'}"
        )
        combo_dir = out_root / tag
        if not args.dry_run:
            combo_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(_REPO / "tools" / "external_arena.py"),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(combo_dir),
            "--games", str(int(args.games)),
            "--parallel-games", str(int(args.parallel_games)),
            "--device", str(args.device),
            "--seed", str(int(args.seed) + idx * 1009),
            "--opp-engine", str(args.opp_engine),
            "--our-sims", str(int(sims)),
            "--our-c-puct", str(float(c_puct)),
            "--our-q-weight", str(float(q_weight)),
            "--our-q-clip", str(float(q_clip)),
            "--our-temperature-move", str(float(args.temperature_move)),
            "--pikafish-binary", str(args.pikafish_binary),
        ]
        if args.opp_depth > 0:
            cmd += ["--opp-depth", str(int(args.opp_depth))]
        elif args.opp_movetime_ms > 0:
            cmd += ["--opp-movetime-ms", str(int(args.opp_movetime_ms))]
        elif args.opp_nodes > 0:
            cmd += ["--opp-nodes", str(int(args.opp_nodes))]
        else:
            raise SystemExit("set one opponent strength knob")
        if root_noise:
            cmd.append("--our-add-root-noise")

        print(f"[{idx}/{len(combos)}] {tag}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        t0 = time.monotonic()
        rc = subprocess.run(cmd).returncode
        elapsed = time.monotonic() - t0
        payload = _load_latest_arena_json(combo_dir)
        row = {
            "tag": tag,
            "returncode": int(rc),
            "seconds": elapsed,
            "sims": int(sims),
            "c_puct": float(c_puct),
            "q_weight": float(q_weight),
            "q_clip": float(q_clip),
            "root_noise": bool(root_noise),
        }
        if payload is not None:
            row.update({
                "score_rate": payload.get("score_rate"),
                "our_wins": payload.get("our_wins"),
                "opp_wins": payload.get("opp_wins"),
                "draws": payload.get("draws"),
                "elo_estimate": payload.get("elo_estimate"),
                "json": str(sorted(combo_dir.glob("external_arena_*.json"))[-1]),
            })
        summary.append(row)
        (out_root / "mcts_grid_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"  rc={rc} dt={elapsed:.0f}s score={row.get('score_rate')}", flush=True)

    if args.dry_run:
        print("dry-run complete; no matches launched", flush=True)
    else:
        print(f"DONE: wrote {out_root / 'mcts_grid_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
