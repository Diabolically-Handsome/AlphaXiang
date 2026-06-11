#!/usr/bin/env python3
"""Small V14S night-watch helper.

Wait for a running V14S grid to finish, pick the best completed row, and launch
one follow-up FPU grid around that row.  This is intentionally conservative and
only orchestrates search-side experiments; it never edits model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "v14s_search_grid_summary.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if int(row.get("returncode", 1)) == 0 and row.get("json")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-run-dir", required=True)
    parser.add_argument("--previous-pid", type=int, default=0)
    parser.add_argument("--output-root", default="/home/laure/alphaxiang/v14s_search_tuning")
    parser.add_argument("--min-score-to-continue", type=float, default=0.50)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    previous_run_dir = Path(args.previous_run_dir)
    output_root = Path(args.output_root)
    log_path = previous_run_dir / "auto_continue.log"

    def log(message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"watching previous_run_dir={previous_run_dir} pid={int(args.previous_pid)}")
    while int(args.previous_pid) > 0 and _pid_alive(int(args.previous_pid)):
        time.sleep(float(args.poll_seconds))

    # Give the runner a moment to flush summary files after the process exits.
    time.sleep(5.0)
    rows = _load_rows(previous_run_dir)
    if not rows:
        log("no completed rows found; not launching follow-up")
        return 0

    rows.sort(
        key=lambda row: (
            float(row.get("score_rate", -1.0)),
            -float(row.get("score_stderr", 1.0)),
            float(row.get("avg_plies", 0.0) or 0.0),
        ),
        reverse=True,
    )
    best = rows[0]
    best_score = float(best.get("score_rate", 0.0))
    log(
        "best row: "
        f"tag={best.get('tag')} score={best_score:.3f} "
        f"W-L-D={best.get('our_wins')}-{best.get('opp_wins')}-{best.get('draws')}"
    )
    if best_score < float(args.min_score_to_continue):
        log(f"best score below {float(args.min_score_to_continue):.3f}; not launching FPU follow-up")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"phase1_fpu_d5_black_{stamp}_from_logcpuct"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "V14S_OUT": str(out_dir),
            "V14S_DEVICE": "cuda:0",
            "V14S_GAMES": "1",
            "V14S_OPENING_SUITE": "arena_openings/human_val_opening_suite_v1.json",
            "V14S_MAX_OPENINGS": "4",
            "V14S_GAMES_PER_OPENING": "1",
            "V14S_CPUCT": str(float(best.get("c_puct", 1.45))),
            "V14S_CPUCT_BASE": str(float(best.get("c_puct_base", 1.0))),
            "V14S_CPUCT_FACTOR": str(float(best.get("c_puct_factor", 0.0))),
            "V14S_Q_WEIGHT": "1.0",
            "V14S_Q_CLIP": "1.0",
            "V14S_FPU_ROOT": "0.05,0.10",
            "V14S_FPU_TREE": "0.10,0.20",
            "V14S_SIMS": "8000",
            "V14S_TEMP": "0.02",
        }
    )
    cmd = ["bash", "tools/_run_v14s_phase1_coarse_grid.sh"]
    runner_log = out_dir / "runner.log"
    log(f"launching FPU follow-up: out={out_dir}")
    with runner_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out_dir / "runner.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    log(f"FPU follow-up pid={proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
