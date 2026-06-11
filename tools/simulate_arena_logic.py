#!/usr/bin/env python3
"""Simulate new arena acceptance logic on existing arena JSON files."""
import json
import glob


def main():
    arenas = sorted(glob.glob("/home/laure/alphaxiang/arena_runs/arena_*.json"))
    print(f"{'time':<16} {'step':<7} {'W':>3} {'L':>3} {'D':>3} {'dec_wr':>7} {'accept_new':>12}")
    print("-" * 64)
    for f in arenas:
        with open(f) as fp:
            d = json.load(fp)
        w = d["candidate_win"]
        l = d["champion_win"]
        dr = d["draw"]
        dec = w + l
        dec_wr = w / dec if dec else 0.5
        nd = d["non_draw"]
        score = d["candidate_score_rate"]
        by_score = score >= 0.55 and nd >= 3
        by_dec = dec_wr >= 0.60 and dec >= 10
        accept = by_score or by_dec
        if accept:
            tag = "SCORE" if by_score else "DECISIVE"
        else:
            tag = "reject"
        ts = f.split("/")[-1].replace("arena_", "").replace(".json", "")[-13:]
        print(f"{ts:<16} {d['candidate_step']:<7} {w:>3} {l:>3} {dr:>3} {dec_wr:>7.3f} {tag:>12}")


if __name__ == "__main__":
    main()
