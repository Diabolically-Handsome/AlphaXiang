# PROJECT BRIEF v3 — v13 architecture decision (after q_weight Pareto-trade finding)

**Date:** 2026-05-02
**Audience:** the same external consulting agent (you)
**Context:** revision of v2. v2's "+118 Elo from q_w=2.0 inference fix" was a 50-game lucky sample; panel re-eval shows it's a Pareto trade, not a uniform improvement. The inference-side ceiling is harder than v2 suggested. v13 capacity scaling becomes more justified, not less.
**Tone requested:** keep pushing back. You caught us twice already.

---

## 0. TL;DR

Two corrections from v2:

1. **The "bilateral lemma" (in-dist search-bound, OOD capacity-bound) was already correctly questioned by you in round 2 and falsified by us in round 3.** Replaced by a single absolute-strength ceiling at **~public Elo 2030** for v12 sims=1600.

2. **The "q_weight=2.0 → +118 Elo" claim from v2 was wrong.** It was a 50-game lucky sample. The real effect of q_w=2.0 is a **Pareto trade**: helps weak/dissimilar opponents (Pika d=5 +9pp, Fairy +11pp) but hurts equal-strength opponents (Pika d=3 −17pp). Net panel-mean ≈ +55 Elo, but the d=3 regression makes q_w=2.0 unsuitable for default deployment.

**The new story:**
> v12's absolute ceiling is hard. Inference-side knob tuning trades performance across opponent classes but **does not raise the absolute ceiling**. The only path to uniform Elo gain across opponent classes is capacity scaling.

**What we want from this round:**
1. **Sanity-check the corrected story.** Single hard ceiling + Pareto-trade-on-knobs — is this the right read, or is there something else we're missing?
2. **Updated v13 capacity sizing.** With the inference-side floor now closer to 0, v13's job is harder. Does 150M dense still seem right, or should we plan for 200-250M?
3. **v12.6 finetune — go or skip?** With inference fix giving ~0 Elo guaranteed, training-side fix is the only thing that might rescue v12. Worth 7 days, or skip to v13?

---

## 1. What changed since v2

### 1.1 Pikafish depth ladder (full data)

We did your suggested depth ladder, but on Pikafish (ElephantEye+Wine deadlocked at parallel-games=4):

| Pika depth | v12 score (sims=1600, q_w=1.0) | Implied v12 Elo |
|---:|---:|---:|
| d=3 | 49.25% | ~2050 |
| **d=4** | **20.0%** | ~2010 |
| **d=5** | **8.0%** | ~2025 |
| d=6 | 3.0% | ~2050 (saturating) |
| d=7 | 1.0% | ~2050 (saturated) |
| d=10 (positive control, in-dist!) | 1.0% | floor |

Three reliable anchors triangulate to **v12 absolute Elo ≈ 2030 ± 50**.

### 1.2 The bilateral lemma was wrong

v2 assumed Pika d=10 (in-dist) should be playable while ElephantEye d=10 (OOD) was blocked. Positive control showed **both score 1%**. The "OOD wall" was just absolute strength.

### 1.3 Three-way diagnostic on 2020 d=4 failure positions

| Q | Metric | Result |
|---|---|---|
| Q1 (policy) | pika best in v12 top-5 | 60.8% |
| Q2 (value) | pearson(cp, v) | 0.36 (mid-game 0.16) |
| Q2 (value) | sign correctness | 88.7% |
| Q3 (MCTS choice) | chosen ∈ policy top-1 set | 72.8% |

The value head has **fuzzy magnitude but reliable sign** (88.7%). MCTS itself is fine.

### 1.4 q_weight grid finding (the v2 correction)

Hypothesis from Q2: "value head is fuzzy → reduce its MCTS weight should help."
Result: **opposite**.

| q_weight | Pika d=4 score |
|---:|---:|
| 0.25 | 7% |
| 0.50 | 5% |
| 0.75 | 19% |
| 1.0 (baseline) | 20% |
| 1.5 | 22% |
| 2.0 | 33% (50-game sample), **21% (replicate)** |

Lower q_weight catastrophically removes the value sign signal MCTS needs. Higher q_weight initially seemed to help dramatically (33%) but replicate revealed the 33% was a lucky sample — true effect is +5-7pp on d=4.

### 1.5 q_weight=2.0 panel re-eval (the v2 falsifier)

| Opponent | q_w=1.0 | q_w=2.0 | Δ |
|---|---:|---:|---:|
| Pika d=3 | 49.25% | **32.0%** | **−17pp** |
| Pika d=4 (replicate) | 20.0% | 21.0% | +1pp |
| Pika d=5 | 8.0% | 17.0% | **+9pp** |
| Fairy-SF d=3 | 82.0% | **93.0%** | **+11pp** |

**Pareto trade**: helps stronger / different-style opponents, hurts equal-strength similar opponents. Net panel-mean ≈ +55 Elo, but d=3 regression makes default deployment impossible.

A single q_w=1.5 vs d=3 cell is currently running to test if there's a "safe middle" preserving d=3 strength.

### 1.6 Day 3 pipeline 1-6 complete (training data ready)

Failure-slice training data fully prepared:
- 2841 our-turn positions from 36 d=4 losses + 8 draws
- Oracle value (Pikafish d=12), oracle policy (Pikafish d=8 multipv=5 adaptive temp), teacher_q on hard rows
- Hard mining: 710/2841 hard rows flagged (top 25% disagreement)
- Audit: clean (0 dirty shards, 0 illegal entries)

Awaiting user decision on whether to run Day 3 step 7 (actual finetune from v12 PEAK).

---

## 2. The corrected lemma (cleaner now)

```
Lemma 1 (search-budget unlock):
  Inference-time MCTS sims=800 → 1600 yields +30pp on Pika d=3, +20pp on CNN,
  +12pp on Fairy. Free Elo gain via search calibration alone.

Lemma 2 (absolute capacity ceiling):
  Once sims-saturated, v12 caps at ~public Elo 2030 (Pika d=3 even match).
  Above this, score collapses: d=4=20%, d=5=8%, d=6=3%, d=7+ saturated.
  Same collapse profile in-dist (Pika) and OOD (ElephantEye).

Lemma 3 (search-knob Pareto trade):
  Inference knob tuning (q_weight, c_puct, etc.) cannot raise the absolute ceiling.
  Higher q_weight specializes toward stronger/different opponents at the cost of
  equal-strength performance. No knob configuration uniformly improves all panel cells.
  Therefore inference-side ceiling is at ~+20 Elo for any safe-middle config (best case).
```

These three lemmas together imply: **capacity scaling is the only path to uniform Elo gain across opponent classes.**

---

## 3. v12.6 — substantially weaker case than v2 implied

v2 had v12.6 expected gain at +60-100 Elo (mostly via teacher_q + failure slice).
v3 has revised expectation: +30-60 Elo at best, given:
- Inference side gives ~0 (Pareto-bounded) at safe q_weight, maybe +10-20 if q_w=1.5 holds d=3
- Training side gives +30-50 (per literature) but only on the failure slice positions, with risk of in-dist regression

The total v12.6 envelope is uncertain at +30-80 Elo. **And it costs 7 days of work.**

vs v13 which we expect +200 Elo for ~3 weeks of work. The Elo-per-week ratio is similar but v13 has a much higher ceiling.

**Our updated lean: skip full v12.6 finetune. Ship q_w=1.5 (if it holds d=3) as v12.6-lite, then go to v13.**

---

## 4. v13 capacity sizing — slightly larger?

v2 recommended 150M. With inference floor now revised down, the case for v13 is firmer:

| v13 size | Public Elo target | Crosses Pika d=4 reliably? |
|---|---:|---|
| 80M (2× v12) | ~2150 | No (still below d=4=2250) |
| 150M (4× v12) | ~2250 | **Yes, marginal** |
| **200M (5× v12)** | **~2350** | **Yes, with comfort** |
| 300M (8× v12) | ~2400 | Yes, possibly past d=5 (2450) |

The "marginal" at 150M is concerning given our scaling estimate has a ±50 Elo error band. 200M might be the safer bet for "definitely past d=4". Or commit to 150M with the understanding that v14 (300M) is the next step if v13 lands at 2200.

---

## 5. Specific questions for you

### 5.1 Sanity-check the corrected story
- Single absolute ceiling + Pareto trade — is this read sound, or is there a fourth alternative?
- Specifically the Pareto trade — is this a known phenomenon in MCTS literature? Or is it specific to our broken-but-useful value head?
- Is 200M dense the right answer, or should we go bigger to ensure safe past Pika d=4?

### 5.2 v12.6 — your call
With inference floor revised down, is 7 days of v12.6 finetune still worth it? Or does the data say "skip to v13" cleanly now?

If we DO run v12.6 finetune, what's the single most informative change to make in the training recipe specifically to fix value head magnitude (which Day 1 Q2 identified as the bottleneck)?

### 5.3 Architecture for v13 — is "separate value trunk" still your top pick?
v2 you recommended 2D rel pos bias + separate value trunk/pooling. With the new finding that value head is "noisy but useful binary classifier", does that change the value-trunk priority?

Specifically: if v12's value head is essentially a noisy binary classifier, would a discrete WDL head (W/D/L softmax) be a better v13 design than a scalar value head with sharper τ?

### 5.4 Pareto trade as paper material
The lemma "search-knob tuning is a Pareto trade across opponent classes, not a uniform Elo gain" — is this novel? AlphaZero-likes papers usually report a single panel score; we may be the first to show different inference configs win on different panel cells.

### 5.5 Risk audit
What's the most likely failure mode for v13 = 200M dense, given:
- Same Pikafish d=12 oracle (we haven't strengthened the teacher)
- Same training pipeline (no architectural changes besides scale)
- Same value head architecture (noisy binary classifier)

Is "scaled v12 = scaled noisy classifier, plateau at ~+150 Elo" a real risk?

---

## 6. Reference files (audit anything)

- `memory/v126_day1_lemma_correction.md` — bilateral lemma falsified
- `memory/v126_day1_diagnostic_findings.md` — three-way diagnostic on d=4 failures
- `memory/v126_day2_qweight_finding.md` — q_weight grid + Pareto trade correction
- `memory/v126_day3_pipeline_ready.md` — training data ready, awaiting finetune decision
- `tools/value_policy_diagnostic.py` — new diagnostic tool
- `tools/_run_v126_day23_chain.sh` — Day 2/3 driver script
- `/home/laure/alphaxiang/v126_day3_d4_slice/train/` — failure-slice shards (audit clean)

---

## 7. What we are NOT asking
- Permission. User has approved capacity scaling.
- A complete v13 spec.
- Whether to publish the lemmas. (We will, separately.)

Please tell us anything we're getting wrong this round — wrong assumptions, missed alternatives, places the data does not support the conclusion. Direct, technically-precise feedback is what we need.
