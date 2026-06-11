#!/usr/bin/env python3
"""Print today's arena results with promotion info."""
import json
import glob
from pathlib import Path


def main():
    files = sorted(glob.glob("/home/laure/alphaxiang/arena_runs/arena_20260418_*.json"))
    print(f"{'time':<6} {'c_step':<7} {'ch_step':<7} {'W':>3} {'L':>3} {'D':>3} {'score':>6} {'dec_wr':>7} {'prom':>7}")
    print("-" * 62)
    promos = 0
    for f in files:
        with Path(f).open() as fp:
            d = json.load(fp)
        ts = Path(f).stem.replace("arena_20260418_", "")
        cs = d.get("candidate_step", 0)
        chs = d.get("champion_step", 0)
        w = d.get("candidate_win", 0)
        l = d.get("champion_win", 0)
        dr = d.get("draw", 0)
        s = d.get("candidate_score_rate", 0.0)
        dwr = d.get("decisive_winrate", 0.0)
        prom = d.get("promoted", False)
        if prom:
            promos += 1
        p = "YES" if prom else "-"
        print(f"{ts:<6} {cs:<7} {chs:<7} {w:>3} {l:>3} {dr:>3} {s:>6.3f} {dwr:>7.3f} {p:>7}")
    print("-" * 62)
    print(f"Total: {len(files)} arenas today, {promos} promotions")


if __name__ == "__main__":
    main()
