# v13 Targeted Repair 024 Results

Date: 2026-05-07

## Summary

The strongest checkpoint is still:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`

Targeted repair found useful failure structure, but no replacement checkpoint is ready to ship.

Main conclusion: the d5 weakness is not a simple head-level bug. Value-only and policy-head-only patches can move one anchor, but they create Pareto damage elsewhere. The only direction that looked technically healthy was a full-model anchored refutation micro-run, but its d5 gain did not survive expansion.

## New Code / Interfaces Added

- `tools/verified_failure_slice.py`
  - Builds a narrow first-blunder policy repair slice from failure-analysis JSON.
  - Targets teacher best moves rather than the played losing moves.

- `tools/post_blunder_value_slice.py`
  - Builds value-focused after-blunder states.
  - Used to test whether the model can learn to recognize refuted child positions.

- `xiangqi_train.py`
  - Added `--train-only-policy-head`.
  - Added `--train-only-value-head`.
  - These freeze the trunk and isolate tiny repair probes to policy or value/WDL heads.

## Data Built

- Verified first-blunder policy/regret slice:
  - `/home/laure/alphaxiang/v13_failure_v2_data/verified_first_blunders_teacherq_d12`
  - 69 samples, hygiene clean.

- Post-blunder value slice:
  - `/home/laure/alphaxiang/v13_failure_v2_data/post_blunder_value_d15`
  - 26 samples, hygiene clean.

- After-only value slice:
  - `/home/laure/alphaxiang/v13_failure_v2_data/post_blunder_after_only_value_d15`
  - 13 samples, hygiene clean.

- Broader after-only d5/red-balanced value slice:
  - `/home/laure/alphaxiang/v13_failure_v2_data/post_blunder_after_only_d5red_broader_value_d15`
  - 50 samples, hygiene clean.

- d5 opening refutation oracle-policy slice:
  - `/home/laure/alphaxiang/v13_failure_v2_data/d5_opening_refutation_policy_d10`
  - 200 samples, hygiene clean.

## Repair Arms Tested

### 024b: concentrated first-blunder policy/regret

Checkpoint:
`/home/laure/alphaxiang/training_runs/run_024b_v13_verified_failure_concentrated_from022d18000/snapshots/latest_step18200.pt`

Arena:
`/home/laure/alphaxiang/v13_snapshot_smoke/run_024b_step18200_d5_sims8000_10g/pika_d5/external_arena_20260507_101353.json`

Result: `2W-4L-4D = 40.0%`

Conclusion: offline top-3 movement did not convert to arena strength.

### 024d: post-blunder value-head repair, anchored

Checkpoint:
`/home/laure/alphaxiang/training_runs/run_024d_v13_post_blunder_valuehead_from022d18000/snapshots/latest_step18200.pt`

Conclusion: value MSE improved slightly, but the anchor pulled toward the old wrong value. Not promoted to arena.

### 024e: after-only value-head repair, no anchor

Key results:

- step18200 d4:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024e_step18200_d4_sims6400_10g/pika_d4/external_arena_20260507_131056.json`
  - `7W-1L-2D = 80.0%`
- step18200 d5:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024e_step18200_d5_sims8000_10g/pika_d5/external_arena_20260507_134237.json`
  - `2W-6L-2D = 30.0%`
- step18300 d3:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024e_step18300_d3_sims8000_10g/pika_d3/external_arena_20260507_122038.json`
  - `5W-3L-2D = 60.0%`
- step18300 d4:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024e_step18300_d4_sims6400_10g/pika_d4/external_arena_20260507_124815.json`
  - `2W-4L-4D = 40.0%`
- step18300 d5:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024e_step18300_d5_sims8000_10g/pika_d5/external_arena_20260507_113212.json`
  - `5W-5L-0D = 50.0%`

Conclusion: real Pareto trade. More value repair helps d5 but damages d4; milder repair helps d4 but damages d5.

### 024g: broader d5/red-balanced after-value repair

Partial d5 result for step18400:

- `/home/laure/alphaxiang/v13_snapshot_smoke/run_024g_step18400_d5_sims8000_10g/pika_d5/arena_console.log`
- Stopped after `1W-5L-1D` partial.

Conclusion: broader after-value did not solve d5.

### 024h: d5 opening oracle-policy head repair

Partial d5 result for step18100:

- `/home/laure/alphaxiang/v13_snapshot_smoke/run_024h_step18100_d5_sims8000_10g/pika_d5/arena_console.log`
- Stopped after `1W-6L` partial.

Conclusion: policy-head-only opening repair did not solve d5.

### 024i: full-model anchored refutation micro-run

Checkpoint:
`/home/laure/alphaxiang/training_runs/run_024i_v13_refutation_fullmodel_anchored_from022d18000/snapshots/latest_step18500.pt`

Results:

- d5 6-game screen:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024i_step18500_d5_sims8000_6g/pika_d5/external_arena_20260507_161542.json`
  - `3W-3L-0D = 50.0%`
- d4 6-game screen:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024i_step18500_d4_sims6400_6g/pika_d4/external_arena_20260507_163540.json`
  - `5W-0L-1D = 91.7%`
- d5 expansion, partial:
  - `/home/laure/alphaxiang/v13_snapshot_smoke/run_024i_step18500_d5_sims8000_extra14g_seed2026050714/pika_d5/arena_console.log`
  - Stopped after `1W-4L-2D` partial.

Conclusion: 024i is the healthiest repair direction, and the d4 signal is very strong, but d5 did not survive expansion. Do not ship as v13 replacement yet.

## Technical Interpretation

1. The d5 losses are not random.
   They cluster around repeated opening families and tactical refutations.

2. The model can learn local crisis recognition.
   Value-head-only repair changed after-blunder sign accuracy from about 23% to 92% on the tiny after-only slice.

3. Local crisis recognition alone is not enough.
   The resulting MCTS behavior shifts openings and creates d4/d5 Pareto tradeoffs.

4. Policy-head-only is also not enough.
   Re-ranking d5 opening candidates at the head level did not transfer.

5. The next credible direction is not another tiny patch.
   It should be a system-level refutation curriculum:
   - larger high-quality d5 loss/refutation data,
   - oracle policy + teacher_q + post-blunder value together,
   - full-model training with a strong 022d anchor,
   - frequent d3/d4/d5 smoke gates.

## Recommendation

Do not replace 022d yet.

Recommended next run:

- Build a larger refutation dataset from at least 100-200 d5 games, not 25 loss games.
- Mine only:
  - repeated opening divergence positions,
  - first avoidable drops,
  - post-blunder opponent-to-move refutations.
- Label with:
  - oracle policy depth 10-12,
  - teacher_q depth 12,
  - oracle value depth 15.
- Train from 022d with full model enabled, but anchored to 022d policy/value.
- Stop any candidate unless it passes:
  - d3 not below 55%,
  - d4 not below 60%,
  - d5 above 50% on at least 20 games.

The evidence says v13 has room, but d5 is a curriculum/representation issue, not a one-head repair issue.
