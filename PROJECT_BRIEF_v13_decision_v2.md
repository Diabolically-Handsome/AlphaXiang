# PROJECT BRIEF v2 — v13 architecture decision (corrected after positive control)

**Date:** 2026-05-01
**Audience:** the same external consulting agent (you)
**Context:** this is a **revision** of `PROJECT_BRIEF_v13_decision.md`. The bilateral lemma you correctly pushed back on is now empirically falsified. The corrected picture is cleaner and the v13 question is sharper.
**Tone requested:** keep pushing back. You caught the over-interpretation last round; we want you to do it again.

---

## 0. TL;DR

You were right to call our bilateral search-vs-capacity lemma premature. Following your suggestion to add a **positive control** (v12 vs Pikafish d=10, fully in-distribution strong opponent) and a **proper depth ladder** (Pikafish d=4/5/6/7), the real picture is much simpler:

> **v12 has a single absolute-strength ceiling around public Elo 2030 (95% CI 1980–2080). Both in-distribution (Pika d=10) and OOD (ElephantEye d=10) opponents above ~2300 produce essentially zero score. The ceiling is not OOD-specific — it's just absolute strength.**

This actually makes v13 cleaner: capacity scaling is the lever to push the ceiling up, period. No need for an OOD-specific data curriculum; the same training pipeline at higher capacity should help.

**What we want from this round:**
1. **Sanity-check the corrected lemma.** Single absolute-strength ceiling — does this hold up, or is there a third explanation we're still missing?
2. **v12.6 vs skip-to-v13.** v12.6 training-side fixes (failure-slice + teacher_q + WDL) are upper-bounded at ~+60-100 Elo per literature. Is that worth a week, or should we skip directly to v13?
3. **v13 capacity sizing.** Our calibration says 80-150M params should reach Pika d=4 (~+200 Elo). Is that conservative, optimistic, or about right?
4. **Architectural adds for v13.** Of {2D rel pos, piece-type embed, palace/river attention masks, full from-to policy MLP}, which 1-2 are worth the implementation cost?

---

## 1. What we did since v1 brief — and what falsified the lemma

You suggested running:
- **(a)** ElephantEye depth ladder to find a non-saturated diagnostic point
- **(b)** opening-FEN diversification
- **(c)** value-head/policy diagnostic on failure positions
- **(d)** v12.6 training-side fixes

We did (a) partially (ElephantEye + Wine concurrency stalled — see §3 below) but added something you didn't suggest: **a positive control on Pikafish d=10**. That single experiment killed the bilateral lemma in one shot.

### 1.1 The positive control — the falsifying experiment

| Opponent | Distribution | sims | v12 score | W-L-D |
|---|---|---:|---:|---:|
| ElephantEye d=10 | OOD (alpha-beta) | 1600 | 0.0% | 0-50-0 |
| **Pikafish d=10** | **IN-distribution (NNUE)** | 1600 | **1.0%** | **0-49-1** |

If the ceiling were OOD-specific, Pikafish d=10 (in-dist) should be playable. It isn't. **In-dist and OOD strong opponents both crush us.** The mechanism is not distribution mismatch; it's just absolute strength.

### 1.2 Pikafish depth ladder (the real diagnostic)

Replacing the stalled ElephantEye ladder with Pikafish d=4-7 (50 games each, sims=1600):

| Pika depth | Likely public Elo | v12 score | W-L-D | Implied v12 Elo |
|---:|---:|---:|---:|---:|
| d=3 (prior) | ~2050 | 49.25% | 77-80-43 (200 games) | **2045** |
| **d=4** | **~2250** | **20.0%** | 6-36-8 | **2009** |
| **d=5** | **~2450** | **8.0%** | 2-44-4 | **2026** |
| d=6 | ~2650 | 3.0% | 0-47-3 | 2046 (saturating) |
| d=7 | ~2850 | 1.0% | 0-49-1 | 2052 (saturated) |
| d=10 (prior) | ~2700+ | 1.0% | 0-49-1 | floor |

**Three independent reliable anchors (d=3/4/5) all triangulate to ~2030.** This is now our most confident absolute-Elo estimate.

### 1.3 The strength gradient is steep — no diagnostic sweet spot

Each Pikafish depth step costs us roughly **−200 Elo**:
- d=3 → d=4: −240 Elo
- d=4 → d=5: −185 Elo
- d=5 → d=6: −180 Elo
- d=6 → d=7: −195 Elo

The transition from "v12 wins" (49% at d=3) to "v12 saturated" (1-3% at d=6+) happens entirely in **one depth step**. There is no 25-75% diagnostic depth for v12. The only point with measurable headroom is **d=4 at 20%**, and that's the v12.6 anchor by elimination.

---

## 2. The corrected lemma

### 2.1 Falsified hypothesis (from v1 brief)
~~In-dist search-bound, OOD capacity-bound (bilateral)~~

### 2.2 Current best hypothesis (single-sided)

> **v12 has a single absolute-strength ceiling at ~public Elo 2030 (38.6M params, sims=1600 inference). Doubling MCTS sims (800→1600) lifts panel readings up to this ceiling but cannot push past it. Capacity scaling is the only lever to push the ceiling itself.**

### 2.3 Sub-claims (paper-ready)

1. **Search-bound below ceiling.** Inference MCTS sims=800 was systematically under-utilizing the network's latent capacity. Doubling to 1600 unlocked +30pp on Pika d=3, +20pp on CNN, +12pp on Fairy-SF. This is a "free" Elo gain available to any AlphaZero-style engine that has been training-undersearched at inference time.

2. **Hard capacity ceiling above search-saturation.** Once sims-saturated at 1600, v12 hits a wall at Pika d=3 / public Elo ~2030. Anything above this loses cleanly across multiple opponent types (Pikafish family, ElephantEye, presumed Fairy-SF d=5+).

3. **The "panel ceiling vs capacity ceiling" distinction.** The conventional intuition that "panel saturation = need more capacity" conflates two effects. Search calibration must be exhausted *first*; the residual capacity question is what's left after.

### 2.4 Honest open questions

- We have only Pika and (partial) ElephantEye anchors. Other engine paradigms (Cyclone, OpenChess Xiangqi forks) might behave differently — but we have no reason to expect this would change the absolute-strength interpretation.
- The 200 Elo/depth gradient on Pika is steeper than published "100 Elo/depth" estimates for Stockfish. May be Pikafish-specific selectivity or that we're in the steep part of v12's own curve. Doesn't affect the lemma.

---

## 3. ElephantEye + Wine — operational note

For completeness: we tried `parallel-games=4` against ElephantEye on WSL via Wine. Both d=2 and d=8 cells stalled silently after games 19 and 29 respectively (UCCI pipe deadlock; processes alive, no output). We abandoned ElephantEye in favor of native-Linux Pikafish. The lemma doesn't depend on ElephantEye anyway. If we ever need ElephantEye anchor data, `parallel-games=1` is the workaround.

---

## 4. Realistic v12.6 target (heavily revised)

Your v12.6 plan (failure-slice + teacher_q + adaptive-temp + value-dampened MCTS + 80/20 mixed curriculum) is sound. But the original target ("ElephantEye non-saturated depth +10pp") is invalidated. New target:

| Anchor | current | v12.6 acceptable | v12.6 great |
|---|---:|---:|---:|
| Pika d=3 | 49.25% | ≥55% | ≥60% |
| **Pika d=4** | **20%** | **≥30%** | **≥40%** |
| Pika d=5 | 8% | ≥15% | ≥25% |
| Panel mean (Fairy + CNN + d=1n15) | basis | not drop >5pp | not drop >3pp |

Acceptable = +60-100 absolute Elo gain (training-side ceiling per Lc0-class scaling literature).
Great = +120-150 absolute Elo (very ambitious for fixed 38.6M).

**Concern:** even "great" still leaves us at ~2150 public Elo, well below Pika d=4's ~2250. So v12.6 *cannot* unilaterally cross past Pika d=4. It can move us closer, but v13 capacity is the real lever.

---

## 5. v13 capacity scaling — our current best estimate

Using Lc0-class scaling laws (~+75-100 Elo per 2× params, with architecture inductive bias possibly worth another 30-50% efficiency):

| v13 size | Public Elo target | Train cost | Hardware |
|---|---:|---:|---|
| 80M (2× v12) | ~2150 | 1-2 weeks | consumer GPU |
| **150M (4× v12)** | **~2250 (Pika d=4)** | **2-3 weeks** | **consumer GPU** |
| 300M (8× v12) | ~2400 (Pika d=5 region) | 4-6 weeks | multi-GPU |
| 600M (15× v12) | ~2550 (between d=5/6) | 2 months | A100-class |

We're inclined toward **150M dense** as the v13 target. Reasoning:
- 4× capacity is the biggest jump where consumer hardware still works
- Aiming for Pika d=4 (~+200 Elo) is concrete and measurable
- If 150M doesn't get us there, scale-pull-out is informative on its own
- 300M+ is v14 territory — keep one capacity tier per generation

---

## 6. Specific questions for you

### 6.1 Sanity-check the corrected lemma
Single absolute-strength ceiling at ~2030. Does this hold up under your scrutiny? Or is there a remaining alternative we should consider before committing to v13 capacity scaling? Specifically:

- Could the ceiling be **value-head specific**? (i.e., the value head is bottlenecked but the policy head has more headroom; if we trained a fresh value head on v12 backbone we'd unlock more)
- Could it be **MCTS architecture specific**? (i.e., our specific PUCT formulation is the bottleneck, not network capacity; AlphaZero variants with different selection rules might extract more from the same network)
- Could it be **training-data ceiling rather than capacity ceiling**? (i.e., we've exhausted what Pikafish d=12 oracle can teach; need d=15+ teachers or different supervision)

### 6.2 v12.6 — go or skip?
Two arguments for skipping:
- Best-case v12.6 (+150 Elo) still leaves us 100 Elo below v13 target
- v12.6 takes ~7 days, delaying v13 start
- We already exhausted search side; new data on whether training-side helps comes "for free" if we run v12.6 anyway, but the conclusion may not change v13's design

Two arguments for doing v12.6 first:
- value-head/policy/MCTS three-way diagnostic (Day 2 of v12.6) directly informs v13 architecture (do we need separate value trunk? full from-to policy head?)
- v12.6 surfaces ANY problems with the new shard formats / training pipeline before v13's expensive retrain
- If v12.6 unexpectedly beats expectations (>+150 Elo), v13 timeline is partially relaxed

What's your call?

### 6.3 v13 capacity sizing
- Is **150M dense** about right for "+200 Elo from 38.6M baseline"?
- Should we go **width-favored** (16L × d=768 ≈ 100M) or **depth-favored** (24L × d=512 ≈ 75M)? Xiangqi has small context (90 tokens) but rich tactical patterns
- We're skeptical 80M is enough. Are we wrong?

### 6.4 v13 architectural adds
Pick at most 2 from this list, or argue we should add nothing:
- 2D relative position bias for the 9×10 board
- Piece-type-aware K projections (separate K for pawns/cannons/horses/etc.)
- Palace/river structural attention masks
- Full from-to MLP policy head (replacing current low-rank 256×256 factorization)
- Separate value trunk (current value head is just a linear projection of the trunk)

### 6.5 Training data for v13
- Reuse all v12 shards + add new self-play shards generated by v13 itself?
- Or fresh Stage 1 retrain from scratch on Pikafish d=15 distillation?
- Or warm-start v13 from v12 PEAK weights (with appropriate width/depth interpolation)?

This is mostly an engineering question but the answer affects training time by ~3×.

### 6.6 Training-time sims for v13
v12 trained at sims=800. v12 inference unlock is at sims=1600. Should v13 train at sims=1600?
- Pro: training targets are sharper, model learns from better-quality MCTS rollouts
- Con: 2× training compute per step
- Mid: train at sims=1200 (sweet spot)

### 6.7 Fail-informatively design
What's the single most likely failure mode for our v13 plan, and how do we structure the training to detect it before sinking 3 weeks?

---

## 7. What we are NOT asking
- Permission. The user has approved capacity scaling.
- A complete v13 spec. Direction + 2-3 critical decisions is plenty.
- Whether the lemma is publishable. (We think it is, but it's a separate write-up.)

---

## 8. Reference files

- `memory/v126_day1_lemma_correction.md` — full Day 1 results + lemma correction + capacity scaling table
- `memory/v125_panel_sims1600_full.md` — original (now-falsified) bilateral lemma write-up
- `memory/v12_full_validation.md` — v12 panel at sims=800
- `tools/_run_v126_day1_pika_ladder.sh` — produced the §1.2 depth ladder
- `tools/external_arena.py` — has all the search knobs we tried
- `xiangqi_model.py`, `xiangqi_train.py` — current 38.6M architecture

Please be direct about anything we're getting wrong — wrong assumptions, missed alternatives, places where the data does not support the conclusion.
