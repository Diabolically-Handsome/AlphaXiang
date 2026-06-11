"""Step 1 of root-cause: per-game pattern analysis on the 70-game tournament."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


# C++ termination codes (from xqcpp_ext_hist8_115.cpp / xiangqi_arena consts):
TERM_NAMES = {
    -1: "ongoing",
    0: "checkmate",       # win/lose by mate
    1: "max_plies_draw",
    2: "repetition_draw",
    3: "no_capture_draw",
    4: "perpetual_check_loss",
}


def load_tournament(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tournament_json", type=Path)
    args = ap.parse_args()
    data = load_tournament(args.tournament_json)
    games = data["per_game"]

    print("=" * 72)
    print(f"TOURNAMENT: {data['transformer_checkpoint'].split('/')[-1]} vs {data['cnn_weights'].split('/')[-1]}")
    print(f"Total: {data['transformer_wins']}W - {data['cnn_wins']}L - {data['draws']}D / {data['games']} games")
    print(f"Score rate: {data['score_rate']*100:.1f}%   Decisive winrate: {data['decisive_winrate']*100:.1f}%")
    print("=" * 72)

    # 1. Color breakdown
    print("\n--- COLOR BREAKDOWN ---")
    by_color: dict[str, dict[str, int]] = {
        "red": {"W": 0, "L": 0, "D": 0},
        "black": {"W": 0, "L": 0, "D": 0},
    }
    for g in games:
        side = "red" if g["transformer_is_red"] else "black"
        if g["outcome"] == "transformer_win":
            by_color[side]["W"] += 1
        elif g["outcome"] == "cnn_win":
            by_color[side]["L"] += 1
        else:
            by_color[side]["D"] += 1
    for side, c in by_color.items():
        total = sum(c.values())
        wp = 100 * c["W"] / total if total else 0
        print(f"  Transformer as {side:>5}: {c['W']:2d}W-{c['L']:2d}L-{c['D']:2d}D / {total}  ({wp:.1f}% win)")

    # 2. Termination breakdown
    print("\n--- TERMINATION BREAKDOWN ---")
    term_by_outcome: dict[str, Counter] = {
        "transformer_win": Counter(),
        "cnn_win": Counter(),
        "draw": Counter(),
    }
    for g in games:
        term = TERM_NAMES.get(g["termination_code"], f"unknown_{g['termination_code']}")
        term_by_outcome[g["outcome"]][term] += 1
    for outcome, counts in term_by_outcome.items():
        if not counts:
            continue
        total = sum(counts.values())
        breakdown = ", ".join(f"{name}={n}" for name, n in counts.most_common())
        print(f"  {outcome:>16} ({total} games): {breakdown}")

    # 3. Game length distribution
    print("\n--- GAME LENGTH DISTRIBUTION (plies) ---")
    by_outcome_len: dict[str, list[int]] = {
        "transformer_win": [],
        "cnn_win": [],
        "draw": [],
    }
    for g in games:
        by_outcome_len[g["outcome"]].append(int(g["plies"]))
    for outcome, plies in by_outcome_len.items():
        if not plies:
            continue
        plies.sort()
        median = statistics.median(plies)
        mean = statistics.mean(plies)
        # short = <50, medium = 50-100, long = >100
        short = sum(1 for p in plies if p < 50)
        med   = sum(1 for p in plies if 50 <= p < 100)
        long_ = sum(1 for p in plies if p >= 100)
        print(f"  {outcome:>16}: n={len(plies):2d}  median={median:.0f}  mean={mean:.1f}  "
              f"short(<50)={short}  med(50-99)={med}  long(>=100)={long_}  "
              f"min={min(plies)}  max={max(plies)}")

    # 4. Quick "blowout" detection: did Transformer get crushed in <50 plies often?
    fast_losses = [g for g in games if g["outcome"] == "cnn_win" and g["plies"] < 50]
    fast_wins = [g for g in games if g["outcome"] == "transformer_win" and g["plies"] < 50]
    print(f"\n--- BLOWOUT ANALYSIS (games <50 plies) ---")
    print(f"  Transformer crushed quickly: {len(fast_losses)} games  (fraction of losses: "
          f"{100*len(fast_losses)/max(1,data['cnn_wins']):.1f}%)")
    print(f"  Transformer crushed quickly OPP: {len(fast_wins)} games  (fraction of wins: "
          f"{100*len(fast_wins)/max(1,data['transformer_wins']):.1f}%)")

    # 5. Long-grinding losses (>=100 plies, transformer loss)
    long_losses = [g for g in games if g["outcome"] == "cnn_win" and g["plies"] >= 100]
    long_wins = [g for g in games if g["outcome"] == "transformer_win" and g["plies"] >= 100]
    print(f"\n--- ENDGAME ANALYSIS (games >=100 plies) ---")
    print(f"  Transformer endgame losses (>=100 plies): {len(long_losses)} / {data['cnn_wins']} losses")
    print(f"  Transformer endgame wins   (>=100 plies): {len(long_wins)} / {data['transformer_wins']} wins")

    # 6. Per-game outcome streak analysis
    seq = [g["outcome"][0] for g in sorted(games, key=lambda g: g["index"])]  # T/C/d
    streak_pattern = "".join({"transformer_win": "T", "cnn_win": "C", "draw": ".", }[g["outcome"]]
                              for g in sorted(games, key=lambda g: g["index"]))
    print(f"\n--- OUTCOME SEQUENCE (T=Transformer win, C=CNN win, .=draw) ---")
    # Print in chunks of 10
    for i in range(0, len(streak_pattern), 10):
        chunk = streak_pattern[i:i+10]
        print(f"  games {i+1:2d}-{min(i+10, len(streak_pattern)):2d}: {chunk}")

    # 7. Decisive-game balance: ignore draws
    decisive_T = data["transformer_wins"]
    decisive_C = data["cnn_wins"]
    elo_diff = -400 * (decisive_C - decisive_T) / max(1, decisive_T + decisive_C) if False else None
    # Proper Elo from score
    sr = data["score_rate"]
    if 0.001 < sr < 0.999:
        import math
        elo = -400 * math.log10(1.0 / sr - 1.0)
        print(f"\n--- ELO ESTIMATE ---")
        print(f"  Transformer is approximately {elo:+.0f} Elo relative to CNN best")


if __name__ == "__main__":
    main()
