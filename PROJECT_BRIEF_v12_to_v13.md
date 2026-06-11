# PROJECT BRIEF — v12 → v13 architecture decision

**Date:** 2026-04-30
**Audience:** external consulting agent
**Decision in scope:** whether to scale up the model architecture for v13, and if so, in what direction
**Tone requested:** push back hard, look for what we're missing, don't be polite

---

## 0. TL;DR — what we want from this discussion

We've just finished v12, our 4th specialization cycle in Stage 2. The training loop is now clean (a critical canonicalization bug was exposed and fixed mid-cycle). Internal panel metrics moved very little vs v11; head-to-head vs v11 looks impressive (+271 Elo) but we suspect a substantial fraction of that is same-architecture style exploit, not real strength gain.

**Our working hypothesis:** the 38.6M-parameter Transformer is at architectural ceiling for the current data regime. The natural v13 move is to scale model size (depth × width).

**What we want from you:**
1. **Falsify the ceiling diagnosis.** Is "panel mean flat + same-arch h2h dominant + Pika-d=3 stuck" actually evidence of architectural ceiling, or could there be a non-arch explanation we're missing?
2. **If we do scale,** depth-vs-width tradeoff at the ~38.6M → ~100M jump? Any architectural inductive bias we should add for Xiangqi specifically (2D relative position, piece-type embeddings, palace/river structural priors)?
3. **Are there data-side improvements** that would beat scaling at the same compute budget?
4. **Honest read** on the public-ladder Elo gap: internal panel says ~1995, but ElephantEye d=10 anchor says we're maybe 1500–1800 公开 Elo. Is that a translation issue, an overfitting issue, or evidence that our internal panel composition is too narrow?

---

## 1. Project capsule

- **Game:** Xiangqi (Chinese chess), 9×10 board, 8100-dim move space (`from_square × to_square`).
- **Approach:** AlphaZero-style; Transformer encoder + policy/value heads + MCTS; trained via distillation + pikafish-as-opponent self-play.
- **Two stages:**
  - **Stage 1 (v0–v3):** broad distillation from a CNN teacher + Pikafish bootstrapping. Ended at v3, step 181000. Public Elo vs random ≈ 297 (a sanity check, not a competitive number).
  - **Stage 2 (v4–v12):** specialization cycles, each ~7500–15000 steps continuing from previous PEAK, mixing Pikafish-self-play data with curated distillation shards.
- **Latest checkpoint:** `PEAK_step286000_v12_probe2_score95pct_d1.pt` (38.6M params, ~155 MB).

---

## 2. Current architecture and training stack

### 2.1 Model (38.6M params total — confirmed by `sum(p.numel() ...)`)

| Component | Spec |
|---|---|
| Encoder | 12-layer Transformer |
| `d_model` | 512 |
| FFN dim | 2048 |
| Heads | 8 |
| Tokens | 90 board squares + a few special tokens (turn, history flags) |
| Position encoding | learned absolute, per-token |
| **Policy head** | **low-rank from-to factorization**: project to `from_emb [256]` and `to_emb [256]` separately, outer product gives `90×90` ≈ 8100 logits |
| Value head | scalar via mean-pool + 2-layer MLP, `tanh` |

**Important note about the policy head:** logits over the full 8100 action space are produced, but until v12, training used a plain log-softmax over all 8100. v12 added a **legal mask** so log-softmax is restricted to legal moves per position.

### 2.2 State representation

- 90 squares; each square's token is the embedding of (piece type × side) + position embedding.
- 8-step history is included as concatenated channels per square (so the per-square embedding has access to the last 8 board states).
- **Canonical frame:** for black-to-move positions, the board is mirrored top-to-bottom so "self" is always at the bottom from the model's POV. Move indices and legal indices are also canonicalized to match.
- Full from-to action enumeration: 8100 entries indexed by `(from_sq × 90 + to_sq)`. Most are illegal at any given position; legal moves typically ~30–50.

### 2.3 Training data sources (per cycle)

| Source | Share (typical) | What it gives |
|---|---|---|
| Pikafish-self-play shards | ~40% | (state, MCTS_visits, our move, reward) — model plays a Pikafish opponent at varying depths. v11 added a `d=12`-strength opponent (long thinking time per move). |
| Distillation shards (Pikafish d=12 oracle) | ~50% | (state, oracle_value, oracle_policy_distribution) — labeled by querying Pikafish d=12 with multipv=8, mapping cp scores to probability via `tanh(cp/τ)`-like sharpening. v12 used `τ=50` (sharper) vs v11's `τ=200`. |
| Hard-position-mining shards | ~10% | states where the model's value disagrees most with oracle (v11) and/or where the model's top-1 action is far from oracle's top action (v12 added "policy regret"). |

### 2.4 Loss (post-v12, current)

```
total_loss = value_loss + α_policy * policy_loss

value_loss        = MSE(model_v, oracle_v)   # for distillation shards
                  + MSE(model_v, game_outcome)  # for self-play shards

policy_loss       = (1 - α_oracle) * MCTS_visit_CE + α_oracle * oracle_policy_CE

# Both CE terms use legal-masked log-softmax (added in v12):
# masked_logits = logits.masked_fill(~legal_mask, -1e9)
# log_probs = log_softmax(masked_logits, dim=-1)
```

`α_oracle` defaults to 0.5; `α_policy` defaults to 1.0.

### 2.5 Optimization

- AdamW, lr 2e-4, cosine decay, weight_decay 0.01
- Mixed precision (bf16)
- Batch size ~256, grad accum 4 → effective 1024
- ~7500–15000 steps per cycle, 30-game probe vs Pika d=1+n=0.15 every 7500 steps
- Snapshot every 2000 steps, keep last 5
- 2× RTX (cuda:0/cuda:1) on a single host

### 2.6 Compute / wall time

- 1 cycle ≈ 4–6 hours
- Self-play shard generation runs concurrently on cuda:1 while training runs on cuda:0
- v12 completed 15 cycles in ~80 hours wall

---

## 3. v11 → v12 changes (what was supposed to break the ceiling)

These were prescribed by an external review (you, last round) of v11's `PROJECT_BRIEF`. Three changes shipped:

### 3.1 Legal-masked policy CE
v11 had log-softmax over all 8100 actions, including illegal ones. v12 sets `logits[~legal] = -1e9` before log-softmax. This was identified as the single highest-ROI "basic ML hygiene" fix.

### 3.2 Tighter oracle policy temperature
`τ = 200 cp → 50 cp` in `tools/oracle_policy_labeler.py`'s `tanh(cp/τ)` sharpening. Idea: distill sharper, more decisive policies from Pikafish d=12.

### 3.3 Policy-regret hard mining
Added `--policy-regret-weight` to `tools/hard_position_mining.py`. Combines value disagreement with **policy regret**: positions where the model's top-1 action has noticeably worse oracle-q than the oracle's top-K options get up-weighted. Idea: focus capacity on positions where the model is making strategically bad choices, not just numerically uncertain ones.

### 3.4 The canonicalization bug (silent in v10/v11, fatal in v12)

This is the most important new finding for the discussion.

`oracle_policy_labeler.py:_build_oracle_distribution` was storing **raw Pikafish move indices** for oracle-policy targets, while the rest of the training stack uses the **canonical frame** (mirrored for black-to-move). For red-to-move positions raw == canonical so no problem; for black-to-move ~52/100 oracle target indices pointed at the wrong logit.

**Why silent in v10/v11:** without legal masking, log-softmax over all 8100 still gave a finite (small) log-prob for the wrong action, so the gradient was bounded. Effectively v10/v11 trained with ~50% of oracle policy supervision being noise, but never blew up.

**Why fatal in v12:** with legal-masked log-softmax, the wrong index pointed at a `-1e9` logit → `log_softmax = -∞` → `policy_loss = +∞` → gradient with `±∞` entries → Adam destroyed the weights within 1–2 steps. Cycle 5 sanity probe collapsed 90% → 10% (SANITY HALT).

**Fix:** apply `canonical_action(raw_move, stm_is_black)` per move when building the oracle distribution. Verified empirically: `set(oracle_idxs) - set(legal_idxs)` mismatch dropped from 52/100 → 0/100 on a re-labeled shard.

After the fix, v12 trained cleanly from v11 PEAK.

(Documented in detail in `memory/bug_oracle_policy_canonicalization.md`.)

---

## 4. v12 results

### 4.1 Probe trajectory (vs Pikafish d=1+n=0.15, 30 games each)

| cycle | step | v11 probe | v12 probe |
|---|---:|---:|---:|
| 5 | 262500/272500 | 73.3% | **83.3%** |
| 10 (PEAK) | 270000/286000 | 90.0% (27-2-1, score 93%) | **90.0% (27-0-3, score 95%)** |
| 15 | 277500/293500 | 86.7% | 86.7% |

So v12 reaches the same 90% peak rate as v11 but with **zero losses** at the peak — slightly higher score-rate.

### 4.2 Full panel (50 games each, sims=800)

|                        | v10   | v11   | **v12** | v12 W-L-D | Δ panel-Elo (v12-v11) |
|------------------------|------:|------:|--------:|---|---:|
| Pikafish d=1 + n=0.15  | 82.0% | 91.0% | **93.0%** | 46-3-1 | +54 |
| Pikafish d=3           | 10.0% | 18.0% | 16.0%   | 7-41-2 | −23 (within noise) |
| Fairy-SF d=3           | 34.0% | 51.0% | **70.0%** | 32-12-6 | **+140** |
| CNN best (held-out)    | 88.0% | 91.0% | 79.0%   | 32-3-15 | **−165** |

**Panel-mean panel-Elo:** v11 ≈ +133, v12 ≈ +134 — **essentially flat**.

### 4.3 Head-to-head (50 games each, sims=800)

| Match | Result | Score | Elo Δ |
|---|---|---:|---:|
| **v12 vs v11** | 39-7-4 | **82.0%** | **+271** |
| v12 vs v10 | 43-1-6 | 92.0% | +426 |
| v12 vs v7 | **50-0-0** | **100.0%** | ∞ |

For comparison:
- v11 vs v10: 28-11-11 = 67.0% / +123 Elo
- v11 vs v7: 47-0-3 = 97.0% / +604 Elo
- v10 vs v7: 47-0-3 = 97.0% / +604 Elo

So v12 vs v11 (+271) is more than 2× larger than v11 vs v10 (+123), but panel mean barely moved.

### 4.4 Public-ladder anchor (unchanged from v10/v11)

- **ElephantEye d=10 vs v10:** 0-47-0 (v10 lost every game)
- ElephantEye is a publicly-rated engine, estimated ~1700–1900 公开 Elo
- v11 anchor was on track for similar 0-47 result; user aborted the match
- v12 has not been run against ElephantEye yet, but no reason to expect different
- **Implied absolute Elo:** v10/v11/v12 are probably all in the 1500–1800 公开 Elo range, despite internal panel-Elo ~1990

---

## 5. Honest re-evaluation (the user explicitly asked to cool down and re-read the data)

This is what we believe **is** real and **is not** real after looking again:

### 5.1 Calibrating the panel opponents

| Opponent | Likely public Elo | Diagnostic value |
|---|---|---|
| Pikafish d=1+n=0.15 | ~1300–1600 | **Low** — too weak, saturates. Used as cycle probe, not as serious opponent. |
| **Pikafish d=3** | **~1900–2200** | **High** — the only real-strength absolute calibrator on the panel. Pikafish has aggressive selective extensions (LMR, null-move) so nominal d=3 effective seldepth reaches 8–12 on tactical lines. |
| Fairy-SF d=3 | ~1500–1900 | **Medium** — useful as out-of-distribution engine signal (we never trained against Fairy-SF), but Fairy-Stockfish's Xiangqi NNUE is community-built and weaker than Pikafish's dedicated NNUE. ~200–400 Elo weaker than Pika d=3 at the same nominal depth. |
| CNN best | unknown | **Medium** — same training distribution (we both train on Pikafish-style data), so improvements there don't translate to public ladders, but regressions there are diagnostic of style change. |

**Cross-check that Pika d=3 is calibrated correctly:** v10 lost 0-47 against ElephantEye d=10 (~1800). v10 scored 10% vs Pika d=3 (Elo Δ = -382). If Pika d=3 ≈ 1800, then v10 ≈ 1418, which is consistent with "v10 is around 1500 公开 Elo" — within range of the ElephantEye anchor.

### 5.2 The honest re-read of v12

| Metric | Surface reading | Re-read |
|---|---|---|
| Pika d=1+n0.15 91→93 | "small gain" | Saturated. **0 information.** |
| **Pika d=3 18→16** | "noise" | **Absolute strength flat.** This is the only metric that calibrates to public Elo, and it didn't move. Same for v10 → v11 (+8pp) → v12 (-2pp): the meaningful trend is "stuck around 15%." |
| Fairy-SF 51→70 | "+140 Elo" | Real, but on a weaker reference. v12 generalizes better cross-engine — that's real progress, but it's a quality signal, not a quantity signal. |
| CNN 91→79 | "regression" | Real. 3 losses (vs v11's 1) and many more draws. Possibly v12's sharper policy (τ=50) is brittle against CNN's specific style. Yellow flag, not red. |
| v12 vs v11 +271 Elo | "huge generation gain" | **Likely inflated by same-arch style exploit.** v12's sharper τ specifically punishes v11's softer policy regions. v11 vs v10 was +123 Elo and corresponded to ~+50 public Elo (per panel). Linear extrapolation: v12 vs v11 +271 might be ~+100 public Elo at best, possibly less. |

### 5.3 The pattern

**v10 → v11 → v12** on the absolute calibrators:
- Pika d=3: 10% → 18% → 16%
- ElephantEye d=10: 0% → (estimated 0%) → (not measured but no reason to differ)

**v10 → v11 → v12** on training-similar references:
- Pika d=1+n=0.15: 82 → 91 → 93 (saturating)
- CNN: 88 → 91 → 79 (saturating then regressing)
- h2h gains: +604 (v10/v7) → +123 (v11/v10) → +271 (v12/v11)

**Interpretation:** v11's gains over v10 were partly real (oracle policy distillation actually helped), v12's gains over v11 are mostly architectural-ceiling redistribution (sharper policy, better generalization to one OOD engine, but no absolute strength gain).

This is consistent with the working hypothesis: at fixed 38.6M params, the model has saturated what it can learn from Pikafish d=12 distillation. Bigger or more architecturally tuned models could probably extract more.

### 5.4 What this re-read does **not** rule out

- **Data quality ceiling, not arch ceiling.** Maybe Pikafish d=12 oracle is itself the bottleneck (~2700 Elo teacher). Distilling further from a stronger teacher (e.g., Pikafish d=20+ with longer time) might still help at 38.6M params.
- **Distribution mismatch.** Maybe we just haven't trained against opponents in the 1900–2200 range (where Pika d=3 lives). Curriculum gap rather than capacity gap.
- **Search-time issue.** All evals use 800 sims; Pikafish opponents have effectively unbounded (depth-limited only) thinking. At 800 sims, our own MCTS may be the bottleneck on tactical positions, masking what the network actually knows.

These are the most plausible non-arch explanations we can think of. We may be missing others.

---

## 6. The decision: v13 direction options

Roughly in order of "scope of change":

### Option A — pure scale-up (dense Transformer)
Same arch, more params:
- A1: 20L × d=640, FFN=2560 → ~90M params
- A2: 16L × d=768, FFN=3072 → ~100M params
- A3: 24L × d=512, FFN=2048 → ~75M params (depth-favored)
- A4: 12L × d=768, FFN=3072 → ~75M params (width-favored, easier to fit on memory)

Training cost: roughly 2.5× of v12 (longer per step, may need lower batch size, may need to retrain Stage 1 from scratch instead of warm-starting from v12 PEAK).

### Option B — architectural inductive biases for Xiangqi
Keep ~38.6M params, change structure:
- B1: replace learned absolute pos with **2D relative position bias** (board is intrinsically 2D)
- B2: piece-type-aware attention (separate K projections for "self" vs "opponent" pieces)
- B3: structural attention masks for palace/river regions
- B4: full from-to attention table instead of low-rank factorized policy head (would add ~10M params to the head alone)

### Option C — hybrid arch
- C1: CNN backbone (good for spatial patterns) → Transformer top-stack (good for global reasoning)
- C2: Transformer trunk → CNN head for policy-localization

### Option D — keep arch, improve data/training
- D1: stronger teacher (Pikafish d=20 instead of d=12)
- D2: more diverse opponent curriculum (anti-Fairy-SF data, anti-CNN data)
- D3: RL fine-tune from v12 PEAK against Pikafish d=3 directly (the metric we actually want to move)
- D4: increase sims at training time (800 → 1600) so MCTS-derived targets are sharper
- D5: longer cycles (15 cycles → 30 cycles) at fixed arch

### Option E — combination
Most projects in the AlphaZero-likes literature do A + B together. E.g., scale to ~80M with 2D rel pos and a non-low-rank policy head.

---

## 7. Specific questions for you

Please push back on each of these — we'd rather hear "you're wrong because X" than "yes, do A2."

### 7.1 Diagnosis check
We're calling this an "architectural ceiling at 38.6M params for the current data regime." Is that diagnosis sound?
- Specifically: Pika d=3 stuck at ~15% across 3 generations with substantively different training recipes (oracle value, oracle policy + d=15 opponent, legal mask + tighter τ). Is that better explained by capacity ceiling or by something else (data, search, curriculum)?
- Is there a cheap diagnostic experiment (no full retrain) that would falsify this? E.g., increase sims at eval time to 4× and see if Pika d=3 score moves?

### 7.2 If we scale, depth vs width vs heads
At the ~38.6M → ~100M jump, what's the smarter allocation for Xiangqi specifically?
- Xiangqi has 90 tokens (small context) but rich local structure (piece interactions, palace, river)
- Standard scaling laws favor width, but small-context games sometimes do better with depth
- We have empirical reason to believe the model struggles with deep tactical lookahead (Pika d=3 fail mode is repeated tactical errors). Does that argue depth?

### 7.3 Architectural inductive biases worth adding
- Is **2D relative position bias** worth the implementation cost in our case (90 tokens, learned absolute baseline)?
- Is the **low-rank policy head** (256-dim from-emb × 256-dim to-emb) limiting expressivity? Worth swapping for a full from-to MLP?
- Anything else specific to board games we're missing?

### 7.4 Data-side alternatives we should weigh first
Before committing to scale-up:
- Stronger teacher? Pikafish d=20 instead of d=12?
- Curriculum gap fill? (We've never trained against opponents specifically in the 1900–2200 range.)
- More history? Currently 8 plies of history; AlphaZero used 8 for Go, but Xiangqi has different repetition rules.
- RL fine-tune phase against Pika d=3 directly?

If any of these would beat scale-up at the same compute, we'd much rather do them first.

### 7.5 The internal-vs-public Elo gap
Internal panel says ~1990 Elo, ElephantEye anchor says ~1500–1800. That's a 200–500 Elo gap.
- Translation issue (panel composition is biased toward Pikafish-style play)?
- Real overfitting to our specific opponent distribution?
- Anchor reliability issue (ElephantEye d=10 might be stronger than we estimate)?
- All of the above in unknown proportions?

If a v13 architecture gain shows up on internal panel but not on ElephantEye, that's a failure. How do we structure the v13 evaluation so we know early?

### 7.6 Pitfalls / "things you'd hate to see us do"
What's the most likely way we'd waste 2 weeks scaling to v13 and still see flat absolute Elo?

---

## 8. Reference files (if you want to audit anything)

- `xiangqi_train.py` — trainer, loss assembly, legal mask plumbing (post-v12)
- `xiangqi_model.py` — model definition (12L × 512, low-rank policy head)
- `tools/oracle_policy_labeler.py` — Pikafish d=12 multipv-based oracle policy distillation, post-canonical-fix
- `tools/distillation_generator.py` — distillation shard builder (incl. legal mask payload)
- `tools/pikafish_selfplay.py` — Pikafish-as-opponent shard generator
- `tools/hard_position_mining.py` — value-disagreement + policy-regret active learning
- `tools/external_arena.py` — panel evaluation harness
- `tools/transformer_vs_transformer_arena.py` — head-to-head harness
- `tools/_run_v12_full_validation.sh` — the validation script that produced §4 results
- `memory/bug_oracle_policy_canonicalization.md` — canonicalization bug post-mortem
- `memory/v12_full_validation.md` — full v12 result writeup
- `memory/stage2_version_map.md` — version-by-version comparison table
- `memory/project_lemmas.md` — four paper-worthy lemmas from the project so far

---

## 9. What we are **not** asking

- We're not asking for moral support or validation of v12 results.
- We're not asking for an opinion on whether to write a paper (yes, eventually — that's separate).
- We're not asking for a complete v13 design spec — a clear direction + the top-3 pitfalls is plenty.

Please be direct about anything we're getting wrong — wrong assumptions, missed alternatives, places where the data does not support the conclusion.
