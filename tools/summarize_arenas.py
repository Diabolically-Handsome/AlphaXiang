#!/usr/bin/env python3
"""Summarize all arena JSON files in a directory with Elo estimates."""
import json
import math
import sys
from pathlib import Path


def elo_diff(score_rate):
    if score_rate <= 0.001:
        return float("-inf")
    if score_rate >= 0.999:
        return float("inf")
    return -400.0 * math.log10((1.0 - score_rate) / score_rate)


def main():
    arena_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/laure/alphaxiang/arena_runs")
    files = sorted(arena_dir.glob("arena_*.json"))

    print(f"{'timestamp':<20} {'cand_step':<10} {'W':>3} {'L':>3} {'D':>3} {'score':>6} {'plies':>6} {'Elo':>6} {'prom':>6}")
    print("-" * 80)

    cum_w = cum_l = cum_d = 0
    for f in files:
        with f.open() as fp:
            d = json.load(fp)
        ts = f.name.replace("arena_", "").replace(".json", "")
        cs = d.get("candidate_step", 0)
        w = d.get("candidate_win", 0)
        l = d.get("champion_win", 0)
        dr = d.get("draw", 0)
        score = d.get("candidate_score_rate", 0.0)
        plies = d.get("avg_plies", 0.0)
        elo = elo_diff(score)
        prom = d.get("promoted", False) or d.get("accepted", False)
        cum_w += w
        cum_l += l
        cum_d += dr
        prom_str = "PROMOTED" if d.get("promoted", False) else ("ACCEPT" if d.get("accepted", False) else "-")
        print(f"{ts:<20} {cs:<10} {w:>3} {l:>3} {dr:>3} {score:>6.3f} {plies:>6.1f} {elo:>+6.1f} {prom_str:>6}")

    total = cum_w + cum_l + cum_d
    if total:
        cum_score = (cum_w + 0.5 * cum_d) / total
        cum_elo = elo_diff(cum_score)
        print("-" * 80)
        print(f"CUMULATIVE: {cum_w}W-{cum_l}L-{cum_d}D over {total} games, score {cum_score:.3f}, Elo {cum_elo:+.1f}")


if __name__ == "__main__":
    main()
