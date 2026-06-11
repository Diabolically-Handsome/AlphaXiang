# PROJECT BRIEF — Self-play Failure-Mining + d20-Correction Closed Loop

**Date:** 2026-06-02  **Status:** APPROVED, building.  **Director:** user (vision). **Builder:** Claude.

## Goal
Break geo's play ceiling by mining the model's OWN failures via cheap self-play, confirming
they are real *knowledge* gaps (not search artifacts), correcting them with a d20 teacher,
and looping. Targets **POLICY** (move choice), not value.

## Why this, why now
- **Value is tapped (3 experiments):** combo value-merge LOST; ③-补头 value-head finetune made
  value MAGNITUDE more accurate (MAE −26~40%) but SIGN worse → d6 play 0.417 < geo 0.542;
  q_weight↑ (trust value-sign more) also HURT (0.333 < 0.542). Value tweaks don't move play.
- **Policy is the prime suspect:** v12.6 diagnostic measured policy top-5 MISS = 40% (Pikafish
  best move not in model top-5). We then spent all effort on value; never fixed policy.
- **Static d20 pool exhausted** → need FRESH data = the model's own fresh failures.

## Settled design knobs (user decisions)
| Knob | Value |
|---|---|
| Discovery sims | **3200** (validated: vs d5, 1600=0.281 / **3200=0.594** / 12800=0.625 — 3200 ≈ 95% strength at a fraction of cost; 1600 is crippled = fake failures) |
| Material | **lost games** (loser-side positions) |
| Filter | **d20 best move ∉ model top-3 (at 12800-MCTS) → real knowledge gap, keep** |
| Correction | **d20 labels: policy (top-K, primary) + value (cheap add-on)** |
| Form | **closed loop** (correct → retrain → re-mine with new model, snowball) |

## Pipeline (one cycle)
`① 3200 self-play → ② sample loser positions → ③ filter (12800-MCTS top3 vs d20-best) → ④ d20 label policy+value → ⑤ policy-slice train → ⑥ gate→promote → 🔁`

| Step | Tool | New/reuse |
|---|---|---|
| ① mine | `xiangqi_selfplay.py` (3200 sims) | reuse |
| ② sample | small glue (extract loser positions) | NEW (small) |
| ③ **filter** | adapt `value_policy_diagnostic.py` + reuse self-play's MCTS-visit extraction | **NEW (core)** |
| ④ label | `oracle_policy_labeler` + `oracle_value_labeler` (d20) | reuse |
| ⑤ train | `xiangqi_train.py --train-only-policy-head` (escalate to unfreeze trunk if thin readout underdelivers) | reuse |
| ⑥ gate | `_run_gate.sh` | reuse |
| 🔁 orchestrate | adapt `xiangqi_closed_loop.py` (insert ②③④ between selfplay & train) | NEW (glue) |

**Cost optimization:** merge ③+④ d20 queries — one d20 multipv per position serves both the
filter (is best in top-3?) and the labels (top-K). One d20, two uses.

## Build order (bottom-up, smoke each before assembling)
1. Build ③ filter → smoke on last night's failure positions (do d20's good moves really sit
   outside the model's top-3?).
2. ②④⑤⑥ exist → mini-smoke each interface.
3. Insert into closed_loop → dry-run tiny (dozens of games, 1 cycle) — confirm data flows ①→⑥
   unbroken.
4. Clean dry-run → scale up, launch the snowball.

## First-cycle validation (don't snowball blind)
- Filter accuracy: spot-check kept gaps — d20 move truly outside model top-3.
- Correction lift: gated corrected-model > pre-correction (even slightly). If yes → loop is
  real, snowball. If no → STOP and diagnose (never roll blind).

## Complements, doesn't replace: **SCALE** (still 38.6M). Two legs; both eventually. See
[[selfplay_rl_pivot]], [[bu_tou_endgame_vfix]], [[v126_day1_diagnostic_findings]].

## BUILD STATUS (2026-06-02 ~03:00) — pipeline ①→⑤ VALIDATED end-to-end
Tools in `C:\Users\Laure\`:
- `_run_selfplay.sh` — ① 3200 self-play (out: selfplay_loop_smoke/run_001). Games short decisive mates, decisive=100%.
- `_mine_gaps.py` — ②③④ gap-miner. KEY: `mcts_search(board, net, num_simulations, c_puct, q_weight, q_clip, add_root_noise=False, ...)` returns `(best_move, policy_idxs, policy_probs=visit-dist, root_value)`; `SyncNet` wraps `predict_values_and_logits` into the net iface `{policy_logits(N,8100), value_scalar(N,1)}` CPU float32; `_state_to_fen` (from xiangqi_selfplay) decodes state→FEN; ④ reuses `generate_distill_shard(positions, pool, depth=20, ...)`. First run: 15 loser cands → **6 gaps (40%, echoes v12.6 40% policy-miss)**.
- `_fix_slice.sh` — restructure flat slice → `train/`+`manifest.json` for trainer ingest. **TODO: make _mine_gaps write this directly.**
- `_finetune_policy.sh` — ⑤ `--train-only-policy-head` + geometry flags + `--disable-selfplay-run-quality-gate` + small `--replay-buffer-size`. Smoke PASSED: geometry✓, trainable=263682 (policy head), gap-slice ingested (added_samples=6), policy_loss 0.62→0.59↓, mix 0.32/0.68.

**NEXT:** ⑥ gate (`_run_gate.sh`, reuse) + 🔁 orchestrate (adapt `xiangqi_closed_loop.py`: insert ②③④ between selfplay & train) + first REAL cycle (mine ~100+ gaps → train more steps → gate: does correction lift geo vs original?).
