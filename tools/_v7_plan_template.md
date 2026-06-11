# Stage 2 v7 Plan — Mixed-Opponent Curriculum
*Final version, 2026-04-27, based on full panel evaluation*

## 1. Diagnosis from full panel (50 games each, sims=800)

| Engine (Elo est) | v4 score | v5 score | v6 score | trend |
|---|---|---|---|---|
| Pikafish d=1+n0.15 (~1600) | 18-29-3 = 39.0% | 30-16-4 = **64.0%** 🏆 | 26-22-2 = 54.0% | ↓ from v5 peak |
| Pikafish d=3 (~2200)        |  0-50-0 =  0.0% |  0-49-1 =  1.0% |  1-43-6 = **8.0%** 🏆 | ↑ slow climb |
| Fairy-SF d=3 (~2100)        |  0-50-0 =  0.0% |  0-49-1 =  1.0% |  1-48-1 = **3.0%** 🏆 | ≈ marginal |
| CNN best (~1500)            | 31-16-3 = **65.0%** 🏆 | 29-15-6 = 64.0% | 25-19-6 = 56.0% | ↓ −9pp v4→v6 |

**Weighted Elo (mean of per-engine implied Elo):**
- v4 ≈ 857  *(distorted by 0% floor on hard engines — true value ~1450 with Laplace smoothing)*
- v5 ≈ 1500
- v6 ≈ 1610

**Verdict: v6 is OVERALL strongest by weighted Elo (+110 over v5), but it's a specialization trade.**

The mental model that captures it:

| Distance from training distribution | v5→v6 |
|---|---|
| **Close** (Pikafish d=3, Fairy-SF d=3, both close to v6's d=5 training) | **improved** (+7pp / +2pp) |
| **Slightly off-distribution** (Pikafish d=1+n0.15) | regressed (−10pp) |
| **Totally different paradigm** (CNN AlphaZero-style) | regressed (−9pp) |

So v6 traded breadth for depth at training-difficulty.  Not a wholesale loss — just a narrower competence band.

## 2. v7 Design Goals

1. **Continue from v6** (not roll back to v5 — v6 is stronger on weighted Elo)
2. **Restore breadth** by mixing training opponents across the difficulty range that v5 mastered
3. **Preserve v6's depth gain** by keeping some hard-Pikafish exposure
4. **Use existing held-out panel for evaluation only** (no leakage)
5. **Snapshot every 2K steps** (never lose peaks)

## 3. v7 Recipe

### Starting checkpoint
**v6 peak (step 210K)** — confirmed strongest by weighted Elo.

### Mixed Pikafish opponents per cycle (`--vspika-profile`, 4 profiles)
```
40 games per cycle, weighted as:
  - 30%  d=2 + n=0.20  (12 games — very weak, restore v5's weak-opponent dominance)
  - 30%  d=3 + n=0.15  (12 games — medium-easy, broaden middle band)
  - 25%  d=4 + n=0.15  (10 games — medium, where v5 was strong)
  - 15%  d=5 + n=0.10  ( 6 games — hard, preserve v6's depth gain at lower exposure)
```

Why this distribution: v5's strength on Pikafish d=1+n0.15 (64%) shows the model can handle weak opponents well when trained at d=4.  v6's regression there (54%) suggests v6's training drifted too far from "weak" play.  By including d=2+n=0.20 (very weak) at 30%, we re-anchor weak-opponent skill while keeping enough hard exposure to retain v6's gains.

### Sanity probe (every 5 cycles)
- vs Pikafish d=1+n=0.15, 30 games (matches Stage-1 baseline + matches v6's measured strength)
- Halt threshold: score rate < 35% (v6 baseline is 54%; threshold 35% catches major regression without firing on minor variance)

### Hyperparameters (carry over from v6, no changes)
- our_sims: 800
- vspika-parallel-games: 16
- snapshot-interval-steps: 2000
- train-lr-schedule-max-steps: 350000
- WDL/value loss weights: 1.0 / 0.5
- value_target_scale: 0.9
- reset-buffer-on-first-cycle: ON (transplanted from v6)

### Cycles
- 15 cycles auto-stop
- Auto-halt if probe < 35%
- Auto-halt if subprocess crashes/OOMs

## 4. Implementation status

✅ stage1_driver supports multi-profile vspika via `--vspika-profile NAME:GAMES:DEPTH:NOISE:SIMS`
✅ snapshot saving integrated
✅ sanity probe integrated
✅ cross-game batcher integrated
✅ `--reset-buffer-on-first-cycle` for transplanted checkpoint

**No new code needed — pure config change.**

**Optional follow-ups** (defer to v8+):
- Self-play opponent (model vs N-cycle-old snapshot) — would address CNN-style regression more directly
- ElephantArt integration debug (UCCI deadlock)
- Full panel mid-training validation (every 20 cycles)

## 5. v7 Launch Command

```bash
# 1. Seed run dir with v6 peak
mkdir -p /home/laure/alphaxiang/training_runs/run_011_stage2_v7_mixed
cp /home/laure/alphaxiang/PEAK_step210000_v6_probe2_score65pct_d3.pt \
   /home/laure/alphaxiang/training_runs/run_011_stage2_v7_mixed/latest.pt

# 2. Launch
cd "/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
python tools/stage1_driver.py \
  --cycles 15 \
  --train-device cuda:0 --selfplay-device cuda:1 \
  --vspika-profile easy:12:2:0.20:800 \
  --vspika-profile medium_easy:12:3:0.15:800 \
  --vspika-profile medium:10:4:0.15:800 \
  --vspika-profile hard:6:5:0.10:800 \
  --vspika-parallel-games 16 \
  --train-snapshot-interval-steps 2000 \
  --train-lr-schedule-max-steps 350000 \
  --reset-buffer-on-first-cycle \
  --training-output-dir /home/laure/alphaxiang/training_runs/run_011_stage2_v7_mixed \
  --selfplay-root /home/laure/alphaxiang/selfplay_runs_stage2_v7_mixed \
  --sanity-probe-every 5 --sanity-probe-games 30 \
  --sanity-probe-opp-depth 1 --sanity-probe-opp-noise-ratio 0.15 \
  --sanity-probe-our-sims 800 --sanity-probe-min-winrate 0.35 \
  --seed 20260428
```

Estimated wall time: ~225 min (15 cycles × ~15 min/cycle), ~3.75 hours.

## 6. Expected outcomes

### Success criteria (v7 better than v6 on panel)
- **vs Pikafish d=1+n0.15:** ≥ 60% (recover from v6's 54%, approach v5's 64%)
- **vs CNN best:** ≥ 60% (recover from v6's 56%, approach v5's 64%)
- **vs Pikafish d=3:** ≥ 8% (preserve v6's gain)
- **vs Fairy-SF d=3:** ≥ 3% (preserve v6's gain)
- **Weighted Elo:** ≥ 1700 (~+90 over v6)

### Partial success
- v7 trades differently than v6 — some panel slots up, some down
- Need to reconsider whether mixing 4 Pikafish profiles is enough diversity

### Failure (v7 worse than v6)
- Halt early via sanity probe
- Roll back to v6 peak
- Move to v8 with self-play (different style training data)

## 7. Roadmap context

```
v0  random init        → Elo ~150
v1-v3 (Stage 1)         → Elo ~1387 (vs Pikafish d=1+n0.15 at 21%)
v4 peak (step 196K)     → Elo ~857-1450  (start of Stage 2 panel data)
v5 peak (step 204K)     → Elo ~1500   (ladder up d=3→d=4)
v6 peak (step 210K)     → Elo ~1610   (ladder up d=4→d=5, with specialization tradeoff)
v7 peak (target)        → Elo ~1700+  (mixed curriculum, recover breadth)
v8 (future, optional)   → add self-play, target Elo ~1800+
```

This puts v7 in the **业余 2-3 段**(amateur 2-3 dan, ~Elo 1800-2100) territory at peak.

---

**Status: ready for launch.** Awaiting user wake-up.
