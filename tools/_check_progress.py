"""Quick progress check for a running Stage 2 v2 training."""
import json
import sys
from pathlib import Path

sp_root = Path("/home/laure/alphaxiang/selfplay_runs_stage2_v2")
run_dir = Path("/home/laure/alphaxiang/training_runs/run_004_stage2_v2")

# Per-cycle vspika manifests
print("=" * 72)
print("PER-CYCLE VSPIKA RESULTS (training-time winrate)")
print("=" * 72)
print(f"{'cycle':>5} {'tag':30s} {'W':>3} {'L':>3} {'D':>3} {'win%':>6} {'plies':>6}")
print("-" * 72)
rows = []
for d in sorted(sp_root.glob("stage1_c*_vspika")):
    m_path = d / "manifest.json"
    if not m_path.is_file():
        continue
    try:
        m = json.loads(m_path.read_text())
    except Exception:
        continue
    w = int(m.get("wins", 0))
    l = int(m.get("losses", 0))
    draws = int(m.get("draws", 0))
    total = w + l + draws
    win_pct = 100.0 * w / max(1, total)
    plies = m.get("total_plies", 0)
    avg_plies = plies / max(1, total)
    cycle_num = int(d.name.split("_")[1][1:]) if "_c" in d.name else -1
    rows.append((cycle_num, d.name, w, l, draws, win_pct, avg_plies))

for cycle_num, name, w, l, d, wp, ap in sorted(rows):
    print(f"{cycle_num:>5} {name[:30]:30s} {w:>3} {l:>3} {d:>3} {wp:>5.1f}% {ap:>6.1f}")

# Sanity probe log
probe_log = run_dir / "stage1_logs" / "sanity_probe.jsonl"
if probe_log.is_file():
    print()
    print("=" * 72)
    print("SANITY PROBES (vs Pikafish d=8, 20 games)")
    print("=" * 72)
    print(f"{'cycle':>5} {'step':>7} {'W':>3} {'L':>3} {'D':>3} {'win%':>6} {'score%':>7} {'Elo':>6}")
    print("-" * 72)
    with probe_log.open() as f:
        for line in f:
            r = json.loads(line)
            elo = r.get("elo_estimate", 0)
            elo_str = f"{elo:.0f}" if isinstance(elo, (int, float)) else "?"
            print(
                f"{r['cycle']:>5} {r.get('checkpoint_step','?'):>7} "
                f"{r['wins']:>3} {r['losses']:>3} {r['draws']:>3} "
                f"{r['winrate']*100:>5.1f}% {r['score_rate']*100:>6.1f}% {elo_str:>6}"
            )
else:
    print()
    print("(no sanity probes run yet — first one fires after cycle 5)")

# Checkpoint step
latest = run_dir / "latest.pt"
if latest.is_file():
    import torch
    s = torch.load(latest, map_location="cpu", weights_only=False)
    print()
    print(f"latest.pt step: {s.get('global_step','?')}")

# Aggregate
if rows:
    total_w = sum(r[2] for r in rows)
    total_l = sum(r[3] for r in rows)
    total_d = sum(r[4] for r in rows)
    total_g = total_w + total_l + total_d
    print(f"\nOVERALL training winrate so far: {total_w}W-{total_l}L-{total_d}D "
          f"= {100.0*total_w/max(1,total_g):.1f}% over {total_g} games")
