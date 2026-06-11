# Handoff to v13 Agent

**Date:** 2026-05-02
**From:** Claude (transitioning to Reviewer role per user)
**To:** Next agent who will lead v13 capacity scaling
**Status:** v12.6-micro complete on Pika d=3/d=4; full panel verification pending; v13 design ready for execution

---

## 0. Read these first (in order)

1. `memory/MEMORY.md` — index of all project memory
2. `memory/feedback_no_fabrication_under_correction.md` — **CRITICAL team rule.** One fabrication = removed from team. Never invent quotes, backstory, or attribution.
3. `memory/feedback_team_collaboration_tone.md` — never use harsh language to/about other agents (incl. previous Claude sessions)
4. `memory/v126_micro_full_result.md` — full v12.6-micro cycle results (the most recent finding)
5. `memory/v126_day1_lemma_correction.md` — bilateral lemma was falsified; single absolute strength ceiling story
6. `memory/v126_day2_qweight_finding.md` — q_weight Pareto trade (corrected)
7. `PROJECT_BRIEF_v13_decision_v3.md` — most recent design brief (some claims revised by v12.6-micro result)

---

## 1. Where we are right now

**Best known checkpoint:** `/home/laure/alphaxiang/training_runs/run_017_v126_micro/snapshots/latest_step296000.pt` (v12.6-micro)

**Verified strength (sims=1600, q_w=1.0):**
- vs Pika d=3: 47.0% (50 games)
- vs Pika d=4: 39.0% (50 games) ← **+19pp over v12 PEAK**
- vs Pika d=5/d=6/d=7/d=10: not yet tested
- vs Fairy-SF d=3: not yet tested
- vs CNN best: not yet tested
- vs ElephantEye d=10: not yet tested (anchor — likely still 0)

**Estimated absolute Elo:** ~2080-2150 (vs v12 PEAK ~2030).

**Open question for verification before v13:** does v12.6-micro generalize to d=5/Fairy/CNN, or is it specifically optimized for the d=4 failure slice it was trained on? **Strongly recommend running full panel re-eval as v13 first task.**

---

## 2. The v13 question

User has approved capacity scaling for v13. The two key parameters to lock down:

### 2.1 Capacity sizing
- **80M (2× v12)**: probably reaches ~2150 Elo, marginal vs v12.6-micro
- **150M (4× v12)**: aim for ~2250 Elo (Pika d=4 territory) — first-choice per v3 brief
- **200M (5× v12)**: aim for ~2300 Elo (safer cushion) — external agent's recommendation
- **300M+**: v14 territory

### 2.2 Architecture changes (per external agent's v3-round feedback)

Pick **at most 2** from this list:

1. ✅ **2D relative position bias** for the 9×10 board (zero-init, low cost) — RECOMMENDED
2. ✅ **Separate value trunk / value pooling** (current value head reads from material token only) — RECOMMENDED, especially after Day 1 diagnostic showed value head is the main weakness
3. ⚠ **WDL head added** (replace scalar value with W/D/L distribution; MCTS uses WDL expectation) — MEDIUM priority
4. ❌ Palace/river hard attention masks — NOT recommended (rule prior too rigid)
5. ❌ Piece-type-aware K projections — NOT recommended (input planes already encode this)
6. ❌ Full from-to MLP policy head — NOT recommended unless evidence shows policy is the bottleneck

### 2.3 Training pipeline for v13

External agent's view (v3 brief response):
- **NOT warm-start from v12 PEAK** (width/depth interpolation risky)
- **Random init + reuse v12 clean shards for bootstrap, switch to v13 self-play**
- Hard positions use teacher_q, not full d=15 relabel
- Keep v12 rehearsal to protect in-dist panel
- **Training sims**: default 1200, hard/final cycles 1600 (NOT all 1600 — too slow)
- **Eval always sims=1600** for comparability

### 2.4 Failure modes to avoid (per external agent)

1. **"Scaled noisy classifier"** — same teacher + same head + same search at 200M just amplifies v12's flaws
   - Mitigation: include WDL or value trunk change from day 1
   - Mitigation: include dynamic q_weight gating monitoring during training
2. **In-distribution regression** — 50/50 mix of new opponent class destabilizes existing panel
   - Mitigation: 80/20 in-dist/new ratio at most
3. **Loss curves misleading** — flat training loss does not mean no real learning (v12.6-micro showed this)
   - Mitigation: arena-evaluate snapshots periodically during training, not just at end

---

## 3. v12.6-micro lessons (what worked, what didn't)

### What worked (Path D, training-side)
- Failure-slice extraction from arena losses (v126_day3_d4_slice/, 2841 positions)
- value_loss_weight bumped 4× (0.5 → 2.0)
- teacher_q distillation on hard rows (710/2841 = 25%)
- 80/20 v12-rehearsal/failure-slice mix
- Result: +19pp on Pika d=4 in 10K steps (~3 hours wall)

### What didn't work as expected
- **Path C (dyngate q_weight phase gating)**: only +17 Elo panel-mean vs baseline, hurt Fairy by 6pp. Inference-side knob tuning has narrow Pareto frontier.
- **Loss-based monitoring**: value_loss stayed flat throughout Path D training, would have caused me to wrongly conclude "no improvement" if I hadn't run arena. **Always arena-eval, never trust loss alone.**

### What's pending verification
- v12.6-micro full panel (d=5, Fairy, CNN, d=1n15) — could regress like dyngate did on Fairy
- v12.6-micro vs ElephantEye d=10 — almost certainly still 0/50 (this opponent is just stronger than us)
- v12.6-micro vs Pika d=10 (positive control) — almost certainly still 0/50

---

## 4. Tools you'll inherit

All in `tools/`:

- `external_arena.py` — standard arena harness (use `--our-q-weight 1.0` for v13 baseline)
- `external_arena_dyngate.py` — phase-based q_weight (probably not needed for v13)
- `value_policy_diagnostic.py` — three-way diagnostic (Q1 policy, Q2 value, Q3 MCTS); rerun on v13 mid-training to verify value head improvement
- `arena_failure_slice.py` — extract failure positions from arena JSONs
- `oracle_value_labeler.py` / `oracle_policy_labeler.py` / `action_value_labeler.py` — full v12.5 labeling pipeline
- `hard_position_mining.py` — sample weighting + policy regret
- `shard_hygiene_audit.py` — pre-training data audit (run before any training)
- `fix_canonical_meta.py` — patches canonical_action meta on existing shards (one-shot util)
- `mcts_diagnostic_grid.py` — sims/c_puct/q_weight grid for new checkpoints

---

## 5. User collaboration norms (do not violate)

1. **Honest about uncertainty.** "I don't know" / "this is unverified" / "the data is too thin to conclude" — these are required, not weakness signals.
2. **No fabrication, especially under correction.** When called out, acknowledge directly. No backstory invention.
3. **Equal team treatment.** Other agents (including past Claude sessions) are teammates, not subordinates. Critique technical work; never the agent.
4. **Backup before risky actions.** User strongly prefers "test in new path" over "modify existing", "snapshot before overwrite", "verify destructive command before run".
5. **TPM-style decision style.** User makes the strategic calls. Agent proposes options + tradeoffs + recommendation, then waits for go.
6. **Chinese formal "您" register** in user-facing communication.

---

## 6. What I (the previous agent) am NOT certain about

These are good places for v13 agent to apply fresh eyes:

1. **Is v12.6-micro's d=4 jump generalizable?** I claim ~+50-150 Elo absolute but the 50-game variance is ±13pp. Wider eval needed.
2. **Was the value-loss-weight=2.0 the active ingredient, or was it the failure-slice + teacher_q?** Ablation could disentangle. Worth testing if v13 wants to know which knob matters.
3. **The 200 Elo per Pika depth gradient** I quoted in v3 brief — this was estimated from v12 data. v13 at 150-200M might shift this curve; depth ladder should be re-anchored after v13 finishes.
4. **My panel-mean Elo math** treats all 4 panel opponents as equally weighted. If user values Pikafish d=3 more than Fairy (because public-Elo anchor is Pikafish-class), the weighting should be opinionated. This is a TPM decision, not a technical one.

---

## 7. Cleanup status

Completed by previous agent (me) before handoff:
- All schtasks tasks deleted (no orphan scheduled tasks)
- WSL processes verified clean (no zombie pikafish/training)
- Memory index updated
- All briefs corrected (fabricated quote removed)
- New tools committed to `tools/` (uncommitted in git, user can stage at will)

---

## 8. The closing note

This project has gone from v3 (Stage 1 end, ~Elo 1387 panel-rel) to v12.6-micro (~Elo 2100 absolute) in the time I've been here. The user is the consistent through-line — they've made every strategic decision, caught my over-claims, and protected the team's culture.

When in doubt: defer to user decisions, not your own confidence. The data has surprised both me and the external agent multiple times this cycle. Stay humble.

Good luck with v13.

— Claude (founder-agent, retiring to Reviewer role 2026-05-02)
