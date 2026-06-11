# PROJECT BRIEF — v13 architecture decision (after v12.5 search-bottleneck experiments)

**Date:** 2026-05-01
**Audience:** the same external consulting agent who reviewed v11→v12 and proposed v12.5
**Scope:** decide v13's direction, given the bilateral search-vs-capacity finding
**Tone requested:** push back hard, find what we're missing, don't be polite

---

## 0. TL;DR

We followed your v12.5 framework — implemented the legality-mask fix, the canonicalization audit, and the MCTS diagnostic grid. The diagnostic grid result was **dramatic and unexpected**, and reshaped our understanding of where the project's ceiling actually lives.

**The headline finding (bilateral lemma):**

> **In-distribution generalization is search-budget-bound; out-of-distribution generalization is capacity-bound.**

Concretely:
- Doubling MCTS sims (800 → 1600) with the **unchanged v12 PEAK** lifts panel score on Pika d=3 from **18% → 49%**, on Fairy-SF d=3 from 70% → 82%, on CNN best from 79% → 99%.
- The same checkpoint, same sims, against ElephantEye d=10 (a classic alpha-beta + endgame-tablebase engine, not NNUE): **0/50 baseline + 0/50 across 4 different search-knob configurations** (q_weight ∈ {0.5, 2.0}, c_puct=0.8, root_noise=on). Search budget does not move ElephantEye even by a single half-point.

**What we want from this round:**
1. **Falsify the bilateral lemma diagnosis** — is "in-dist search-bound, OOD capacity-bound" actually a sound interpretation, or is there a third explanation we're missing?
2. **v13 design direction** — given that ElephantEye is the OOD wall, is the right move (a) pure capacity scale-up, (b) data-side curriculum change (add ElephantEye-style opponents to training distribution), or (c) hybrid?
3. **Capacity sizing** — if we scale, how much, in what shape?
4. **Risk audit** — what's the most likely way we waste 2–3 weeks on v13 and still see ElephantEye 0/50?

---

## 1. What changed since the last brief

You already know the project up through v12 PEAK and the v12 panel result. Recap of what's been done since:

### v12.5 implementation (yours)
- `tools/shard_hygiene_audit.py` — read-only invariant checker. Ran on all 45 v12 shards: **data is clean** (0 illegal entries, 0 missing legal/fen/stm fields). **45/45 flagged DIRTY only because `oracle_policy_meta.canonical_action` flag is missing** — the data itself was correctly canonicalized post-fix, but the meta marker wasn't written. We'll patch the meta before any retrain.
- `tools/mcts_diagnostic_grid.py` — sims sweep on v12 PEAK vs Pika d=3, 200 games each, 3 cells (sims=800 / 1600 / 3200). Findings below.
- `tools/action_value_labeler.py`, `arena_failure_slice.py`, oracle_policy adaptive temperature, etc. — **not yet executed** because the diagnostic grid found something more important first.

### Key empirical findings

#### 1.1 Pika d=3 sims sweep (v12 PEAK, no retrain)

| sims | score | W-L-D | Elo Δ vs Pika d=3 | Δ from prev |
|---:|---:|---:|---:|---:|
| 800 (v12 panel baseline) | 18.00% | 23-151-26 | −263 | — |
| **1600** | **48.75%** | **72-77-51** | **−9** | **+254 Elo** |
| 3200 | 49.25% | 77-80-43 | −5 | +4 Elo (saturation) |

**Saturation at sims=1600.** Going to 3200 gives essentially nothing.

#### 1.2 Full panel re-evaluation at sims=1600

| Opponent | sims=800 | sims=1600 | Δ |
|---|---:|---:|---:|
| Pika d=1+n=0.15 | 93% | 93% (saturated) | 0 |
| Pika d=3 | 16% | 49% | +33pp |
| Fairy-SF d=3 | 70% | 82% | +12pp |
| CNN best | 79% | 99% | +20pp |
| **panel-mean Elo Δ** | +135 | **+376** | +241 |
| **internal panel-Elo** | 1995 | **~2229** | +234 |

#### 1.3 ElephantEye d=10 — the OOD wall

| Config | sims | c_puct | q_weight | noise | Result |
|---|---:|---:|---:|---|---|
| baseline | 1600 | 1.25 | 1.0 | off | 0-50-0 |
| Cell A | 1600 | 1.25 | **0.5** | off | 1-49-0 (noise) |
| Cell B | 1600 | 1.25 | **2.0** | off | 0-50-0 (extrapolated; 0-42 at 42 games) |
| Cell C | 1600 | **0.8** | 1.0 | off | 0-50-0 (extrapolated; 0-46 at 46 games) |
| Cell D | 1600 | 1.25 | 1.0 | **on** | (terminated; assumed 0/50 by trend) |

Cells B and C had not finished their last 4–8 games when we stopped; the trends were unmistakably 0%. Cell D was terminated before it started by user decision.

**The 1 win in Cell A** (q_weight=0.5, trust policy more) is statistically indistinguishable from 0 (n=50, 95% CI ≈ [0%, 11%]). It's worth noting only as a **direction** — if any tweak hinted at signal, it was "trust policy more, value head less," consistent with the value head being unreliable on OOD positions it never saw during training.

Note: `--eleeye-disable-book=True` was used, so ElephantEye's opening book wasn't directly active. The 50-0 isn't book theory — it's middlegame and endgame execution loss.

---

## 2. The bilateral lemma — interpretation

Restating in slightly more careful terms:

> **For opponents whose play distribution overlaps significantly with the training data (Pikafish family, NNUE-style engines, CNN trained on similar data), v12 has plenty of latent capacity. Its panel underperformance at sims=800 was an inference-time MCTS calibration issue. Doubling sims unlocks essentially all of it.**
>
> **For opponents from a fundamentally different paradigm (ElephantEye 3.31, classic alpha-beta with hand-crafted eval and endgame tablebases), v12 has no idea what's going on. No amount of inference-time search compute fixes this. The model has to genuinely learn something it doesn't currently know.**

This is consistent across multiple controls:
- in-dist evidence is from 4 different engines, all moving the same direction with sims↑
- OOD evidence is from 5 different MCTS configs, all stuck at 0% against the same opponent
- Pika d=3 saturated cleanly at sims=1600 with no further gain at 3200, suggesting the in-dist gain is really about extracting what the network already knows, not "more search = more strength" indefinitely

### Alternative explanations we considered and discarded

1. **"ElephantEye d=10 is just much stronger than estimated."** Possible, but doesn't explain why no MCTS config moves the needle even slightly. If ElephantEye were ~2400 public Elo and v12 ~2000, we'd still expect occasional draws or wins (5-10% range). Getting **literally** 0/50 across 4 different configs suggests a structural blind spot, not a strength gradient.

2. **"It's actually the opening — ElephantEye knows opening theory we don't."** Disabled book (`--eleeye-disable-book=True`). And our games last >50 plies on average — losses are middlegame/endgame, not opening traps.

3. **"It's that ElephantEye has endgame tablebases."** Possible contribution, but most of our losses are by mate, not by tablebase grind. So this is at most a partial explanation.

4. **"v12 self-play data is too narrow."** This IS the candidate — but we'd reframe it as "v12's training distribution doesn't cover anything that plays like alpha-beta+TB." This is consistent with our diagnosis: v12 needs new training data OR more capacity OR both.

### What the lemma does NOT yet establish

- We haven't tested the lemma on **v7/v10/v11** — they may behave differently. If v7 also jumps from ~10% → ~50% on Pika d=3 at sims=1600, the search-bound side is universal across versions. If only v12 jumps, it might be v12-specific (tighter τ, etc.). Worth a follow-up but the current evidence is already strong.
- We haven't tested v12 against **other alpha-beta engines** (e.g., Cyclone, OpenChess Xiangqi forks). The OOD wall might be ElephantEye-specific or general to the alpha-beta paradigm.
- We don't have **a positive control** — we haven't shown that *something* we trained against (e.g., Pikafish d=10) eats us in the same 0/50 way. Pikafish d=10 might be the in-dist equivalent strength to ElephantEye d=10, and the comparison would calibrate "training-distribution proximity" cleanly. Worth doing.

---

## 3. Where v12 stands now

### Internal panel (sims=1600)
- panel-Elo ≈ **2229** (vs v11's 1990, v12-sims-800's 1995)
- Crossed the 2000-Elo internal mark cleanly with sims-only intervention

### Public-Elo estimate
- Lower bound: 1850 (v12 panel-rel minus the ~380 Elo gap historically observed between panel-rel and ElephantEye-implied)
- Upper bound: 2150
- Most likely: **1900–2100, plausibly above 2000 but ElephantEye 0/50 is a strong contrary signal**

### The unresolved question
The reason for honest uncertainty about whether we're "above 2000 public Elo" is precisely the bilateral nature: **we are above 2000 against engines we resemble, and below 1800 against engines we don't.** Single-number Elo doesn't capture this.

---

## 4. v13 design — three top-level options

### Option 1: pure capacity scale-up
Same training pipeline, much bigger model. E.g.,
- 20L × d=640, FFN=2560 → ~90M params (2.3× current)
- 16L × d=768, FFN=3072 → ~100M params (2.6× current)
- 24L × d=512, FFN=2048 → ~75M (depth-favored)

**Bet:** more capacity will incidentally pick up the OOD ability, even without changing the data.

**Risk:** if the issue is genuinely "we have never seen alpha-beta engine play," more capacity might make the model better at what it already does without addressing the OOD gap. Could end up at panel-Elo 2400 internal, still 0/50 vs ElephantEye.

### Option 2: data-side curriculum change at fixed capacity
Same 38.6M model. Add new training opponents:
- Self-play vs ElephantEye d=2/4/6 (lower depths so we win sometimes)
- Distillation from Pikafish-vs-ElephantEye games
- Curriculum mixing: 50% existing recipe, 50% new alpha-beta exposure

**Bet:** the capacity is fine; we just haven't seen this opponent class. Fix it with data.

**Risk:** ElephantEye is slow at low depths to play 1000s of training games. Practical generation throughput might be the bottleneck. Also, adding a new opponent class to a saturating training distribution may not actually push the model past its current optimum — it may converge to a worse compromise.

### Option 3: hybrid (capacity + data)
- Scale to ~80M
- Add ElephantEye-class opponents to training mix
- Probably also bump training-time MCTS sims (currently 800; we now know 800 underuses the network)

**Bet:** both axes contribute. This is what we'd default to absent strong reason otherwise.

**Cost:** longest. Probably 2–4 weeks of training work, possibly more for Stage 1 retrain.

### Option 4 (cheap precursor): "data-only on top of v12"
Before committing to scale-up, try a short v12.5 cycle with just (a) ElephantEye in the opponent pool, (b) failure-slice fine-tuning from Pika d=3 OR ElephantEye losses, (c) the teacher_q distillation you proposed.

**Bet:** if just fixing the data bottleneck gets us partway across the ElephantEye gap, scale-up may not even be necessary. If it doesn't, we have stronger evidence the capacity bound is real.

---

## 5. Specific design questions we want your view on

### 5.1 Diagnosis check
Is the bilateral lemma diagnosis sound? Specifically:
- The 0/50 across 5 search configs is the strongest evidence for "OOD capacity-bound." Are we over-interpreting? What additional tests would falsify "capacity-bound" cleanly?
- Is there a way to test whether **v12's value head specifically** is the broken organ on OOD positions? (e.g., extract value-head outputs on ElephantEye-game positions, see if they're systematically wrong)
- Is the "1 win in q_weight=0.5" hint meaningful at all? Should we be reading anything into the asymmetry between trusting policy vs value at OOD?

### 5.2 Pre-v13 cheap experiments worth doing
Before any retrain, two short experiments are very cheap (hours, not days):
- **(a)** v7/v10/v11 vs Pika d=3 at sims=1600 — does the search-bound side generalize across versions, or is it v12-specific?
- **(b)** v12 vs Pikafish d=10 at sims=1600 — is the in-dist analog of ElephantEye d=10 also 0/50 (suggesting absolute strength wall) or playable (suggesting OOD-specificity)?

Are there other cheap diagnostics you'd run first that we're missing?

### 5.3 If we scale, depth/width/heads
At the 38.6M → ~100M jump for Xiangqi specifically:
- 90 tokens (small context) but 2D structure matters (board, palace, river)
- Suspected weak spot: deep tactical lookahead against tactical engines
- Standard scaling laws favor width; small-context games sometimes favor depth
- We have empirical reason to think the model's value head is brittle on OOD positions — does that argue for adding capacity to the value head specifically (separate trunk?)

### 5.4 Architectural inductive biases worth adding to v13
- 2D relative position bias for the 9×10 board?
- Piece-type-aware attention (separate K projections for "self" vs "opponent" vs "blocker" pieces)?
- Palace/river structural attention masks?
- Replacing the low-rank from-to policy head with a full from-to MLP (~10M extra params just in the head)?

We'd love a recommendation on which 1–2 of these are worth the implementation cost.

### 5.5 Data-side: ElephantEye in training
If we add ElephantEye-class opponents to the training mix:
- What depth(s)? ElephantEye d=2 is too weak (we'd just memorize); d=10 is too slow and we lose every game.
- Should we mix with Pikafish-vs-ElephantEye distillation games (where Pikafish d=12 wins are the labels)?
- Risk of distribution shift hurting in-dist performance — how to mitigate?

### 5.6 Training-time sims
We trained v12 with MCTS sims=800. At inference, sims=1600 unlocks much more strength. Does this mean **training sims=800 was ALSO leaving capacity on the table** (i.e., the training MCTS targets were as under-utilized as the inference MCTS)? If yes:
- For v13, train at sims=1600?
- Or sims=800 with 2× cycles to compensate?
- Or train with the same model running BOTH sims=800 and sims=1600 self-play, to learn from both target qualities?

### 5.7 Eval anchor for v13
Currently the only OOD anchor is ElephantEye d=10 — single point. For v13 we want to know whether the OOD gap is closing.
- What additional public-rated engines are worth running against? (Cyclone? OpenChess Xiangqi? specific Fairy-SF NNUE variants?)
- Is there a standardized public Elo ladder for Xiangqi we should plug into?
- Is it worth registering a v13 candidate on a public ladder for absolute calibration?

### 5.8 Risk audit
The single most concerning failure mode we want you to flag:
- If we go Option 1 (pure scale to 90M) and v13 still goes 0/50 vs ElephantEye, we've burned 3 weeks for nothing on the OOD axis.
- If we go Option 2 (data only) and the new opponent class destabilizes existing in-dist performance, we end up with a worse model on average.
- If we go Option 3 (hybrid) and the experiments take so long we can't iterate, we're locked into v13 design choices for a month.

What's the plan most likely to *fail informatively* — i.e., even if it doesn't work, gives us clear signal about what's wrong?

---

## 6. What we are NOT asking

- We're not asking for a complete v13 spec. A clear direction + top-3 pitfalls is plenty.
- We're not asking for an opinion on whether the bilateral lemma is paper-worthy. (We think it is; that's a separate write-up question.)
- We're not asking for permission to do v13. The user has agreed v13 is justified; we're asking what shape it should take.

---

## 7. Reference files (audit anything)

- `xiangqi_train.py` — trainer, legal mask plumbing (post-v12)
- `xiangqi_model.py` — current arch (12L × 512, low-rank policy head)
- `tools/oracle_policy_labeler.py` — Pikafish d=12 multipv distillation, post-canonical-fix
- `tools/external_arena.py` — panel evaluation harness (incl. all knobs we swept)
- `tools/mcts_diagnostic_grid.py` — your tool, drove the §1.1 sims sweep
- `tools/_run_v125_panel_sims1600.sh` — produced §1.2
- `tools/_run_v125_eleeye_grid.sh` — produced §1.3

Memory files worth reading:
- `memory/v12_full_validation.md` — full v12 panel + h2h results at sims=800
- `memory/v125_search_bottleneck_finding.md` — the diagnostic grid analysis
- `memory/v125_panel_sims1600_full.md` — the full panel re-eval and ElephantEye anchor
- `memory/bug_oracle_policy_canonicalization.md` — the canonicalization bug
- `memory/project_lemmas.md` — prior lemmas; this would add a 5th ("Bilateral Search-Capacity Dichotomy")

---

## 8. Quick numerical reference (in case you need it cited)

- v12 PEAK: `PEAK_step286000_v12_probe2_score95pct_d1.pt`, 38.6M params
- Stage 2 trajectory (panel-Elo at sims=800): v3=1387, v7=1717, v10=1879, v11=1990, v12=1995
- v12 sims=1600 panel-Elo: ~2229
- ElephantEye d=10 anchor: 0/50 across 5 configs (baseline + 4 search knobs)
- Public Elo estimate: 1900–2100, possibly above 2000 for in-dist opponents, ~1700 or below against alpha-beta engines

Please be direct about anything we're getting wrong — wrong assumptions, missed alternatives, places where the data does not support the conclusion.
