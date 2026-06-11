"""Read all panel evaluation JSONs + the existing CNN duel results, build the
'progression-vs-panel' table that lets us see whether v4→v5→v6 is generalizing
or just shuffling specialization.

For each (version, engine) combo, we record W/L/D, score rate, and a relative
Elo (using the standard score-to-Elo conversion).  Then we compute a weighted
average across the panel — that's our 'real Elo' progression."""
from __future__ import annotations

import json
import math
from pathlib import Path

PANEL_BASE = Path("/home/laure/alphaxiang/arena_runs/full_panel")

# Hard-coded existing CNN duel data (already collected at sims=800, 50 games).
EXISTING_CNN = {
    "v4": {"wins": 31, "losses": 16, "draws": 3, "games": 50},
    "v5": {"wins": 29, "losses": 15, "draws": 6, "games": 50},
    "v6": {"wins": 25, "losses": 19, "draws": 6, "games": 50},
}

# Engine slug -> human label
ENGINE_LABELS = {
    "pika_d1n15":  "Pikafish d=1+n0.15",
    "pika_d3":     "Pikafish d=3",
    "fairy_d3":    "Fairy-SF d=3",
    "cnn":         "CNN best",
}

# Approximate Elo of each panel engine (rough, used only for weighted-avg framing).
# These are estimates from prior measurements; uncertainty ±150 Elo.
ENGINE_ELOS = {
    "pika_d1n15": 1600,
    "pika_d3":    2200,
    "fairy_d3":   2100,
    "cnn":        1500,
}


def score_to_elo_diff(score: float) -> float:
    """Standard score -> Elo difference (relative to opponent)."""
    if score <= 0.001:
        return -2000.0
    if score >= 0.999:
        return 2000.0
    return -400.0 * math.log10((1.0 - score) / score)


def load_arena_json(version: str, engine_slug: str) -> dict | None:
    d = PANEL_BASE / f"{version}_{engine_slug}"
    if not d.exists():
        return None
    files = sorted(d.glob("external_arena_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def get_cnn_score(version: str) -> dict:
    rec = EXISTING_CNN[version]
    total = rec["wins"] + rec["losses"] + rec["draws"]
    return {
        "wins": rec["wins"],
        "losses": rec["losses"],
        "draws": rec["draws"],
        "score_rate": (rec["wins"] + 0.5 * rec["draws"]) / total,
        "games": total,
    }


def main() -> None:
    versions = ["v4", "v5", "v6"]
    engine_slugs = ["pika_d1n15", "pika_d3", "fairy_d3", "cnn"]

    # Collect data
    table: dict[tuple[str, str], dict] = {}
    for version in versions:
        for slug in engine_slugs:
            if slug == "cnn":
                table[(version, slug)] = get_cnn_score(version)
            else:
                rec = load_arena_json(version, slug)
                if rec is None:
                    table[(version, slug)] = None
                else:
                    table[(version, slug)] = {
                        "wins": rec["our_wins"],
                        "losses": rec["opp_wins"],
                        "draws": rec["draws"],
                        "score_rate": rec["score_rate"],
                        "games": rec["games"],
                    }

    # Print per-engine table
    print("=" * 90)
    print("v4 / v5 / v6  vs PANEL  —  per-engine score rate")
    print("=" * 90)
    print(f"{'engine (Elo est)':<28} | {'v4':>15} | {'v5':>15} | {'v6':>15} | trend")
    print("-" * 90)
    for slug in engine_slugs:
        opp_elo = ENGINE_ELOS[slug]
        label = f"{ENGINE_LABELS[slug]} (~{opp_elo})"
        row_cells = []
        scores = []
        for v in versions:
            d = table.get((v, slug))
            if d is None:
                row_cells.append("       (missing)".ljust(15))
                scores.append(None)
            else:
                cell = f"{d['wins']:2d}-{d['losses']:2d}-{d['draws']:2d} {d['score_rate']*100:4.1f}%"
                row_cells.append(cell.ljust(15))
                scores.append(d["score_rate"])
        # Trend annotation
        valid = [s for s in scores if s is not None]
        if len(valid) >= 2:
            delta = valid[-1] - valid[0]
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "≈")
            trend = f"{arrow} {delta*100:+.1f}pp v4→v6"
        else:
            trend = "(insuff)"
        print(f"{label:<28} | {row_cells[0]} | {row_cells[1]} | {row_cells[2]} | {trend}")

    # Weighted Elo: each engine contributes equally for now.  We compute the model's
    # implied Elo against each opponent (opp_elo + score_to_elo_diff(model_score)),
    # then average across engines per version.
    print()
    print("=" * 90)
    print("WEIGHTED ELO  (mean of per-engine implied Elo, equal weights)")
    print("=" * 90)
    elo_table: dict[str, dict] = {}
    for v in versions:
        per_engine = {}
        per_engine_elos = []
        for slug in engine_slugs:
            d = table.get((v, slug))
            if d is None:
                continue
            implied_elo = ENGINE_ELOS[slug] + score_to_elo_diff(d["score_rate"])
            per_engine[slug] = implied_elo
            per_engine_elos.append(implied_elo)
        if per_engine_elos:
            mean_elo = sum(per_engine_elos) / len(per_engine_elos)
            elo_table[v] = {"per_engine": per_engine, "mean": mean_elo, "n": len(per_engine_elos)}
            print(f"  {v}: per-engine implied Elo = "
                  + ", ".join(f"{ENGINE_LABELS[s].split(' ')[0]}={e:.0f}" for s, e in per_engine.items())
                  + f"  →  weighted mean = {mean_elo:.0f}  (n={len(per_engine_elos)})")
        else:
            elo_table[v] = None
            print(f"  {v}: no data")

    # Progression
    print()
    print("=" * 90)
    print("PROGRESSION SUMMARY")
    print("=" * 90)
    if all(elo_table.get(v) for v in versions):
        v4_elo = elo_table["v4"]["mean"]
        v5_elo = elo_table["v5"]["mean"]
        v6_elo = elo_table["v6"]["mean"]
        print(f"  v4 peak (step 196K) Elo ≈ {v4_elo:.0f}")
        print(f"  v5 peak (step 204K) Elo ≈ {v5_elo:.0f}    Δ vs v4 = {v5_elo - v4_elo:+.0f}")
        print(f"  v6 peak (step 210K) Elo ≈ {v6_elo:.0f}    Δ vs v5 = {v6_elo - v5_elo:+.0f}")
        print()
        print(f"  Ladder net delta v4→v6: {v6_elo - v4_elo:+.0f} Elo")
        if v6_elo > v5_elo > v4_elo:
            print("  Verdict: monotonic improvement across ladder ✓")
        elif v6_elo < v5_elo < v4_elo:
            print("  Verdict: monotonic regression ↓ — ladder is harming generalization")
        elif v5_elo > v4_elo and v6_elo < v5_elo:
            print("  Verdict: peaked at v5, v6 regressed — v5 is current best generalist")
        else:
            print("  Verdict: non-monotonic — see per-engine breakdown for the story")
    else:
        print("  (incomplete data — at least one version missing)")


if __name__ == "__main__":
    main()
