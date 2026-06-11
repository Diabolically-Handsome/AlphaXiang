# V12.8 FullPika Root-Retune

Date: 2026-05-25

## Decision

Stop treating the V12/V13 problem as a sequence of tiny checkpoint repairs.

The precision probes were useful, but they now look like diagnostics rather
than a product path:

- policy-head root-regret probes repaired offline bad roots, but did not reduce
  `6400`-sim MCTS bad-root count;
- root verifier + mate guard can save concrete positions, but it is slow and
  still needs several safeguards;
- both results point to the same mechanism: the model usually has the good move
  in candidate space, but training did not make root-level ranking robust.

So V12.8 should be a clean retune from V12 PEAK, not a continuation of old
V12.7 and not a V13/V14-style architecture expansion.

## Starting Point

- base checkpoint:
  `/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt`
- do not resume old V12.7:
  `/home/laure/alphaxiang/training_runs/run_020_v127_fullpika_curriculum`
- new train root:
  `/home/laure/alphaxiang/training_runs/run_022_v128_fullpika_root_retune_from_peak`
- new data/eval root:
  `/home/laure/alphaxiang/v128_fullpika_root_retune`

## Core Change

Old V12.7 was not a clean FullPika takeover:

- the new FullPika-labeled data was only about 10% of training mix;
- labels were generated after same-cycle training;
- teacher-Q/pairwise/root-regret losses were off;
- final d6 labels arrived too late to affect training.

V12.8 changes the order and ratio:

1. Generate real V12-vs-Pika d5/d6 arena games.
2. Extract true root positions from those games.
3. Rerun V12 MCTS at `6400` sims and build root candidates.
4. Label candidate children with FullPika/Pikafish:
   - root candidate discovery: `depth=20`, `MultiPV=8`;
   - child evaluation target: `depth=20`.
5. Export root-regret shards with legal masks, teacher-Q, bad-move labels.
6. Only then train.

No shard enters training until it is fully labeled by this FullPika policy.  A
shallower audit may be used only as a pipeline smoke/provisional diagnostic,
not as training fuel.

## Data Mix

Target mix after the root-regret buffer is full:

- 70% FullPika/root-regret selfplay-root data;
- 30% old human/anchor data to prevent broad forgetting.

This is achieved by setting:

- `--bootstrap-human-floor 0.30`
- replay buffer size equal to the root-regret sample count

The point is to make FullPika/root-regret data genuinely dominant, instead of
being a small delayed correction.

## Training Mode

V12.8 is not a policy-head surgical patch.

Default probe trains the full V12 model from V12 PEAK with low learning rate:

- optimizer reset: yes
- LR: `1e-5` initially
- root-regret teacher-Q pairwise: on
- bad-move suppression: on
- teacher-Q listwise: small, optional
- policy/value/WDL losses: kept but reduced, so human anchor still matters
- anchor checkpoint: V12 PEAK
- anchor policy KL/value MSE: on

This is a retune, not random reinitialization.

## Gates

Nothing reaches arena until offline gates pass.

Required gates:

1. Shard read smoke:
   - teacher-Q present;
   - legal masks present;
   - bad-move labels present;
   - losses finite.

2. Short train smoke:
   - root-regret shard ingested;
   - `mix` shows selfplay/root data active;
   - `n_pairwise` and `n_bad_suppress` nonzero.

3. Offline policy/root-regret eval:
   - bad-root safe-top improves;
   - new non-bad regressions are bounded.

4. `6400`-sim root-MCTS gate:
   - bad roots lower than V12 PEAK baseline;
   - catastrophic roots do not increase;
   - missing-candidate does not increase.

5. Arena only after gate:
   - Pika d5 black-side 12 openings;
   - if d5 improves without mate-loss increase, then Pika d6.

## Current Evidence

Pipeline smoke passed:

- static checks passed for the runner and root-regret tools;
- existing V12 PEAK root-regret JSONL converted to a trainer shard:
  `66` train samples, `6` bad roots, `1` catastrophic root;
- shard read smoke confirmed teacher-Q, legal masks, bad-move labels, and finite
  pairwise/suppression losses;
- 2-step GPU train smoke confirmed the shard enters the selfplay buffer on the
  second poll:
  `mix=0.30/0.70`, `n_teacher_q=45`, `n_pairwise=41`,
  `n_bad_suppress=5`.

Rule clarification from the user:

- “FullPika” means both root discovery and child labeling use full depth.
- Shallower labels may be used only for smoke/provisional pipeline checks.
- Shallower labels must not enter formal export, train smoke, train probe, or
  any candidate checkpoint.
- The runner enforces `pika_root_depth >= 20` and `pika_child_depth >= 20`
  before training/export.

Baseline V12 PEAK, d5 black-side 12 openings at `6400` sims:

- `2W-7L-3D / 12 = 29.2%`
- mate-heavy failure profile

Root verifier partial paired run:

- baseline 12-opening completed;
- gated run was stopped at 8/12 after the strategy decision changed;
- partial gated result showed local rescue behavior but not enough to justify
  continuing the slow verifier path as the main route.

This supports using verifier/audit results as training fuel, not as the final
runtime solution.

New V12.8 collection, V12 PEAK black-side, `6400` sims:

- Pika d5:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena/d5/external_arena_20260525_020243.json`
  - `2W-5L-5D / 12 = 37.5%`
  - terminations: `7` mate, `1` max, `4` no-capture
  - average plies: `132.1`
- Pika d6:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena/d6/external_arena_20260525_030334.json`
  - `1W-9L-2D / 12 = 16.7%`
  - terminations: `10` mate, `2` no-capture
  - average plies: `104.8`

Interpretation: d5 still shows real resilience, while d6 produces a thick
mate-heavy failure distribution.  This is exactly the kind of training fuel the
old V12.7 curriculum did not let dominate.

FullPika d20/d20 root audit, first 120-position batch:

- audit:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/root_decision_audit_d5d6.json`
- depths:
  - root MultiPV: `d20`
  - child eval: `d20`
  - V12 MCTS replay: `6400` sims
- counts:
  - bad roots: `7 / 120`
  - catastrophic roots: `6 / 120`
  - missing-candidate: `0 / 120`
  - ranking failure: `7 / 120`
  - Q inversion: `6 / 120`
- shard:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/data/root_regret/shard/train/shard_00000.pt`
  - train samples: `94`
  - bad roots: `7`
  - catastrophic roots: `6`
  - manifest has `fullpika_ok=true`

Interpretation: with both root and child at d20, the failure class remains
clean: true candidate-missing is not the main issue; high-impact failures are
root ranking / Q inversion.  This is the first shard allowed to enter V12.8
training.  Earlier `d10/d12` and `d14/d20` artifacts are provisional only and
marked do-not-train.

Probe A, 500-step full-model retune from V12 PEAK:

- checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022_v128_fullpika_root_retune_from_peak/probe_a_full_model/latest.pt`
- training:
  - `500` steps, `286000 -> 286500`
  - root-regret mix entered correctly: `mix=0.30/0.70`
  - pairwise and bad-move losses were active
- caution:
  - human-val total worsened from the V12 PEAK region to `2.5426`, so this is
    not publishable without arena confirmation.
- offline policy/root-regret eval on all labeled roots:
  - repaired bad roots: `7 / 7`
  - repaired catastrophic roots: `6 / 6`
  - new non-bad regressions vs anchor: `0`
- holdout-only eval:
  - non-bad roots: `17`
  - new regressions vs anchor: `0`
- `6400`-sim MCTS root gate, same 120 roots, d20/d20 verification:
  - V12 PEAK baseline: `7` bad, `6` catastrophic, `0` missing-candidate
  - Probe A: `2` bad, `1` catastrophic, `0` missing-candidate

Interpretation: Probe A gives the first real search-level positive signal, but
it is still same-distribution root validation.  It must pass d5 arena before it
can be considered genuinely stronger than V12 PEAK.

Probe A d5 arena, Pika d5 black-side, `6400` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_a/d5/external_arena_20260525_060311.json`
- score: `2W-4L-6D / 12 = 41.7%`
- comparison to V12 PEAK same config:
  - V12 PEAK: `2W-5L-5D / 12 = 37.5%`
  - Probe A: `2W-4L-6D / 12 = 41.7%`
- positive conversions:
  - opening 2: mate loss -> draw
  - opening 4: short mate loss -> win
  - opening 5: mate loss -> draw
  - opening 10: mate loss -> draw
- regression:
  - opening 7: V12 PEAK win -> Probe A loss

Probe A failed-opening rerun, `25600` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_a/d5_failures_25600/external_arena_20260525_072801.json`
- only the four Probe A `6400`-sim losses were rerun:

| original opening | opening id | 6400 result | 6400 plies | 25600 result | 25600 plies |
|---:|---|---|---:|---|---:|
| 0 | `val_shard_00070_1342` | loss | 119 | loss | 145 |
| 6 | `val_shard_00012_2126` | loss | 109 | win | 128 |
| 7 | `val_shard_00012_2844` | loss | 141 | loss | 225 |
| 9 | `val_shard_00023_3515` | loss | 245 | loss | 115 |

Interpretation: the user's high-sims hypothesis is partially supported.  One
failed opening flips from loss to win, and two losses are substantially delayed,
but one loss worsens.  Search budget helps, but it does not replace training or
mate-risk repair.

Expanded FullPika d20/d20 audit, 300 roots:

- audit:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/root_decision_audit_d5d6.json`
- positions: `300`
- bad roots: `11`
- catastrophic roots: `5`
- missing candidate: `0`
- ranking failures: `11`
- Q inversions: `10`
- deep top-K verifier would help: `11`
- candidate-union verifier would help: `0`
- exported training shard:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/data/root_regret/shard/train/shard_00000.pt`
  - roots written: `298`
  - train roots: `253`
  - holdout roots: `45`
  - samples: `239`
  - bad roots: `11`
  - catastrophic roots: `5`
  - `fullpika_ok=true`

Probe B, conservative 500-step full-model retune from V12 PEAK:

- checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022b_v128_fullpika_root_retune_conservative_from_peak/probe_a_full_model/latest.pt`
- training:
  - `500` steps, `286000 -> 286500`
  - learning rate `2e-6`
  - anchor policy KL `0.20`
  - anchor value MSE `0.10`
  - root-regret mix entered correctly: `mix=0.30/0.70`
- caution:
  - human-val total was `2.5632`, so arena/root gates remain mandatory.
- offline eval on all labeled roots:
  - bad safe-top repairs: `11 / 11`
  - catastrophic repairs: `5 / 5`
  - new non-bad regressions vs V12 anchor: `2 / 272 = 0.74%`
- holdout-only eval:
  - non-bad roots: `44`
  - new regressions vs V12 anchor: `0`
- same-input `6400`-sim root gate on 84 roots extracted from the 300-audit:
  - V12 PEAK baseline: `9` bad, `5` catastrophic, `8` Q inversions
  - Probe B: `4` bad, `3` catastrophic, `4` Q inversions

Probe B d5 arena, Pika d5 black-side, `6400` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d5/external_arena_20260525_092238.json`
- score: `4W-2L-6D / 12 = 58.3%`
- comparison:
  - V12 PEAK: `2W-5L-5D / 12 = 37.5%`
  - Probe A: `2W-4L-6D / 12 = 41.7%`
  - Probe B: `4W-2L-6D / 12 = 58.3%`

Probe B failed-opening rerun, Pika d5, `25600` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d5_failures_25600/external_arena_20260525_094953.json`
- only the two Probe B `6400`-sim losses were rerun:

| original opening | opening id | 6400 result | 6400 plies | 25600 result | 25600 plies |
|---:|---|---|---:|---|---:|
| 6 | `val_shard_00012_2126` | loss | 61 | win | 48 |
| 7 | `val_shard_00012_2844` | loss | 151 | win | 222 |

Interpretation: this strongly supports the high-sims hypothesis for the current
Probe B d5 failure set.  Both failed openings flip from loss at `6400` sims to
win at `25600` sims.

Probe B d6 pressure sample, Pika d6 black-side, `6400` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games/external_arena_20260525_101514.json`
- score: `1W-3L-2D / 6 = 33.3%`
- interpretation:
  - better-shaped than the original V12 PEAK d6 baseline (`1W-9L-2D / 12`),
    but not a d6 breakthrough.
  - use this as a pressure sample, not as a final d6 claim.

Probe B d6 failed-opening rerun, Pika d6, `25600` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_failures_25600/external_arena_20260525_141402.json`
- only the three Probe B d6 `6400`-sim losses were rerun:

| original opening | opening id | 6400 result | 6400 plies | 25600 result | 25600 plies |
|---:|---|---|---:|---|---:|
| 2 | `val_shard_00043_2220` | loss | 201 | draw | 300 |
| 4 | `val_shard_00028_3427` | loss | 119 | win | 100 |
| 5 | `val_shard_00028_0000` | loss | 61 | draw | 238 |

Interpretation: this strongly supports a d6 high-sims effect.  The three
failed openings become `1W-0L-2D` at `25600` sims.  This does not prove d6 is
fully conquered, but it shows the failures are not fixed only by more training;
search budget materially changes the result.

Probe B d6 root audits, FullPika d20/d20:

- all-result 120-root audit:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/probe_b_d6_6games_root_audit_d20d20.json`
  - bad roots: `1 / 120`
  - catastrophic roots: `1 / 120`
  - missing candidate: `0`
  - Q inversion: `1`
- loss-only dense 180-root audit:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/probe_b_d6_loss_only_dense_root_audit_d20d20.json`
  - bad roots: `11 / 180`
  - catastrophic roots: `5 / 180`
  - missing candidate: `0`
  - ranking failures: `11`
  - Q inversions: `8`
  - prior/visit failures: `3`
  - Pika root horizon failures: `3`
- exported loss-only dense root-regret:
  - all candidates:
    `/home/laure/alphaxiang/v128_fullpika_root_retune/data/root_regret/probe_b_d6_loss_only_dense_d20d20_all.jsonl`
  - selected/refuted:
    `/home/laure/alphaxiang/v128_fullpika_root_retune/data/root_regret/probe_b_d6_loss_only_dense_d20d20_selected_or_refuted.jsonl`

Interpretation: the d6 losses still contain repairable root-ranking failures,
but they are concentrated in the losing trajectories.  Missing-candidate remains
at zero, so the next repair batch should keep focusing on root ranking / Q
trust / horizon gating rather than candidate expansion.

Probe C, d6 dense root-regret continuation from Probe B:

- checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022c_v128_fullpika_root_retune_d6_dense_from_probe_b/probe_c_d6_dense/latest.pt`
- data:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/data/root_regret/shard_probe_c_d6_dense/train/shard_00000.pt`
  - combined roots: `478`
  - train samples: `412`
  - bad roots: `22`
  - catastrophic roots: `10`
- training:
  - starts from Probe B
  - `250` steps, `286500 -> 286750`
  - lower LR `1e-6`
  - stronger anchor to Probe B
- offline result:
  - combined data safe-top repairs: `19 / 22`
  - catastrophic repairs: `9 / 10`
  - d6 dense safe-top repairs: `8 / 11`
- MCTS root gate on d6 loss-only dense:
  - Probe C `6400`: `12` bad, `6` catastrophic
  - Probe C `12800`: `11` bad, `3` catastrophic

Interpretation: Probe C did not beat Probe B at `6400` root gate, but doubling
search to `12800` sharply reduced catastrophic errors and selected mean regret
(`565cp -> 247cp`).  This supports the search-budget hypothesis, but Probe C is
not a better checkpoint than Probe B.

Probe D, child-position value-only diagnostic:

- new converter:
  `tools/v13_child_value_audit_to_shard.py`
- child-value shard:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/data/child_value/shard_probe_d_child_value/train/shard_00000.pt`
  - child samples: `432`
  - sources: the 300-root d20/d20 audit plus d6 loss-only dense d20/d20 audit
  - training rows are selected/teacher-best child positions only
- checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022d_v128_child_value_from_probe_b/probe_d_child_value/latest.pt`
- training:
  - starts from Probe B
  - `250` steps, `286500 -> 286750`
  - `--train-only-value-head`
  - policy, teacher-Q pairwise, and bad-move suppression all disabled
- root gate:
  - Probe D d6 dense `6400`: `10` bad, `4` catastrophic
- d6 arena, Pika d6 black-side, `6400` sims:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_d/d6_6games/external_arena_20260525_183718.json`
  - `1W-4L-1D / 6 = 25.0%`

Interpretation: direct value-head child repair is not safe in this form.  It
slightly improves the dense root gate but worsens real d6 play versus Probe B.
Freeze Probe D; keep Probe B as the current best checkpoint.  Child-value data
is still useful as a diagnostic or future sidecar/gate input, but not as a
direct value-head overwrite.

Probe B scalar vs WDL value-source diagnostic:

- trajectory audit:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/offline_eval/probe_b_d6_bad_scalar_vs_wdl_trajectory.json`
- input: the `11` d6 loss-only dense bad roots from Probe B.
- search: `6400` sims, `c_puct=1.25`, `q_weight=1.0`.
- result:
  - scalar value source:
    - repaired known bad roots: `1 / 11`
    - mean known top regret: `7277.5cp`
  - WDL value source:
    - repaired known bad roots: `4 / 11`
    - mean known top regret: `5348.4cp`
- trajectory classes:
  - `horizon_mate`: `5`
  - `early_q_inversion`: `3`
  - `late_q_flip`: `3`
- WDL arena spot check:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_wdl_6400/`
  - stopped after `0W-2L-1D` because it was clearly worse than scalar on the
    same opening prefix.

Interpretation: WDL is a useful diagnostic/risk signal and is less bad than
scalar on the known d6 bad roots, but global `--our-value-source=wdl` is not
safe.  The current issue is likely scalar/Q calibration plus horizon/mate risk,
not simply "switch scalar to WDL".

Probe B d6 full 6-opening rerun, Pika d6 black-side, `25600` sims:

- result:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_25600/external_arena_20260525_212325.json`
- score: `1W-2L-3D / 6 = 41.7%`

| opening | opening id | 6400 result | 6400 plies | 25600 result | 25600 plies |
|---:|---|---|---:|---|---:|
| 0 | `val_shard_00070_1342` | draw | 196 | draw | 159 |
| 1 | `val_shard_00070_2381` | win | 118 | loss | 45 |
| 2 | `val_shard_00043_2220` | loss | 201 | draw | 150 |
| 3 | `val_shard_00043_0124` | draw | 272 | draw | 290 |
| 4 | `val_shard_00028_3427` | loss | 119 | win | 100 |
| 5 | `val_shard_00028_0000` | loss | 61 | loss | 79 |

Interpretation: `25600` improves d6 overall (`33.3% -> 41.7%`) and converts two
of the three scalar-6400 losses, but it is not monotonic: one scalar-6400 win
turns into a short mate loss, and one loss remains a loss.  This is not enough
to justify d7/51200 yet.  Current best checkpoint remains Probe B; next work
should focus on scalar/WDL disagreement as a conditional risk signal plus
horizon/mate-specific gates, not on direct value-head overwrite.

Probe B d6 scalar+WDL shadow disagreement gate smoke, Pika d6 black-side,
`6400` sims:

- main search: scalar value, `6400` sims, `c_puct=1.25`, `q_weight=1.0`
- shadow search: WDL value, `6400` sims, black-side only
- verifier: Pika child depth `16`, top-6 scalar/WDL union, no root mate guards
- status: **non-formal smoke only**.  These runs used d16 verifier and therefore
  must not enter training, acceptance, or release decisions.  They are kept only
  as trigger-shape diagnostics.  Formal verifier/audit/label runs must use
  d20 whenever the experiment is described as FullPika.
- ordinary gate m600:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_shadow_wdl_gate_6400_m600/external_arena_20260525_222050.json`
- ordinary gate m300:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_shadow_wdl_gate_6400_m300/external_arena_20260525_231232.json`

| config | score | events | games with events | key changes vs scalar 6400 |
|---|---:|---:|---:|---|
| scalar baseline | `1W-3L-2D / 6 = 33.3%` | 0 | 0 | reference |
| shadow gate m600 | `2W-2L-2D / 6 = 50.0%` | 2 | 2 | opening 5 flips `loss -> win` |
| shadow gate m300 | `2W-2L-2D / 6 = 50.0%` | 5 | 2 | same score, more late replacements in opening 2 |

Opening-level comparison:

| opening | id | scalar 6400 | m600 gate | m300 gate | scalar 25600 |
|---:|---|---|---|---|---|
| 0 | `val_shard_00070_1342` | draw 196 | draw 196 | draw 196 | draw 159 |
| 1 | `val_shard_00070_2381` | win 118 | win 118 | win 118 | loss 45 |
| 2 | `val_shard_00043_2220` | loss 201 | loss 195 | loss 223 | draw 150 |
| 3 | `val_shard_00043_0124` | draw 272 | draw 272 | draw 272 | draw 290 |
| 4 | `val_shard_00028_3427` | loss 119 | loss 119 | loss 119 | win 100 |
| 5 | `val_shard_00028_0000` | loss 61 | win 166 | win 166 | loss 79 |

Important event notes:

- m600 performs only two replacements.  Opening 5 is the clean success:
  at ply 19, `b9c9 -> e7e8`, verified child eval improves from about `+542cp`
  opponent-pov to `+26cp`, and the game flips from short loss to win.
- opening 2 is not solved.  m600 replaces one mate-risk move at ply 171
  (`d5f6 -> d5e7`, `+20000cp -> +866cp` opponent-pov) but the game still
  loses.  m300 adds three more late mate-risk escapes and only extends the
  loss to 223 plies.
- opening 4 has no gate events in either m600 or m300, despite scalar 25600
  flipping it to a win.  This suggests a separate missing trigger, not merely
  a margin issue.

Interpretation: as a smoke, scalar/WDL disagreement plus child verification is
a useful trigger-shape signal, but the result is not formal because the verifier
was d16.  The next valid run must rerun the same candidate gate with d20.  Also,
ordinary margin lowering alone is unlikely to be sufficient; the d16 smoke
suggested that opening 2 needs repeated mate/horizon auditing and opening 4
needs stricter replacement safety when the original is mate-risk.

Probe B d6 scalar+WDL shadow gate, formal d20 follow-up:

- code change:
  - added `--our-verifier-max-wait-s` so formal d20 child searches can wait
    instead of silently falling back to shallow validation;
  - added `--our-shadow-verifier-ordinary-min-original-cp` to block tiny
    low-risk ordinary replacements.
- strict m200/safe300 d20, opening 4 only:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_opening4_shadow_wdl_gate_rank_ambig_6400_m200_safe300_d20_top12_wait3600/external_arena_20260526_035451.json`
  - result: loss, 69 plies, `0` replacements.
  - interpretation: late rescue is too late; top-12 d20 still finds no safe
    replacement once the position is already badly damaged.
- m50/safe300 d20, opening 4 only:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_opening4_shadow_wdl_gate_rank_ambig_6400_m50_safe300_d20_top12_wait3600/external_arena_20260526_040714.json`
  - result: win, 72 plies, `4` early replacements:
    `g5h5 -> g5f5`, `e7c7 -> e7a7`, `d4b4 -> b6b5`,
    `b5a5 -> b5b4`.
  - interpretation: the user hypothesis is supported.  Some losses are already
    seeded by earlier small root-ranking/value errors; a last-moment mate
    rescue cannot recover them.
- m50/safe300 d20, first two openings:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_2games_shadow_wdl_gate_rank_ambig_6400_m50_safe300_d20_top12_wait3600/external_arena_20260526_051739.json`
  - result: `0W-0L-2D / 2`.
  - opening 1 regressed from baseline win to draw because an early low-risk
    replacement at ply 11 changed the plan:
    `h3h6 -> c9e7`, verified improvement only `54cp`
    (`70cp -> 16cp` opponent-pov).
- m50/min150/safe300 d20, first two openings:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_2games_shadow_wdl_gate_rank_ambig_6400_m50_min150_safe300_d20_top12_wait3600/external_arena_20260526_054900.json`
  - result: `1W-0L-1D / 2`.
  - blocking ordinary replacements when the original child eval is below
    `150cp` restores the opening 1 win and avoids the low-risk plan change.
- m50/min150/safe300 d20, opening 4 only:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_opening4_shadow_wdl_gate_rank_ambig_6400_m50_min150_safe300_d20_top12_wait3600/external_arena_20260526_061209.json`
  - result: draw, 227 plies, `10` replacements.
  - interpretation: the risk floor prevents one known false positive, but the
    online gate remains brittle.  It can trade a short loss for a draw, but it
    can also over-edit a promising line.
- m100/min150/safe300 d20, opening 4 only:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_opening4_shadow_wdl_gate_rank_ambig_6400_m100_min150_safe300_d20_top12_wait3600/external_arena_20260526_063134.json`
  - result: loss, 69 plies, `0` replacements.
  - interpretation: m100 is too conservative for this failure.  The useful
    corrections are often in the `50-100cp` range, but that range is also
    where false positives appear.
- m50/min150/safe300 d20, 6-opening partial run:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_shadow_wdl_gate_rank_ambig_6400_m50_min150_safe300_d20_top12_wait3600_p2/`
  - stopped after three completed games because opening 3 regressed from
    baseline draw to mate loss.
  - observed prefix: opening 1 win, opening 0 draw, opening 3 loss.

Interpretation: formal d20 confirms the mechanism, but not the online gate as
a product setting.  The repair signal is real: early d20-verified small
corrections can reverse an otherwise forced-looking loss.  However, direct
online replacement is fragile because `50cp` child-eval differences can change
strategic plans and introduce new losses.  The better next use of this signal
is training/calibration fuel: teach the model/search which root-Q estimates are
untrusted, instead of shipping a slow d20 gate that edits moves in live play.

Probe B d6 25600 loss-window audit, FullPika d20/d20:

- sparse smoke on the three d6 `25600` losses:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/probe_b_d6_25600_losses_root_audit_d20d20_smoke8.json`
  - sampled `8` roots, found `0` bad roots.  This was too sparse to explain
    the mate losses.
- mate-window audit around the losses:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/audit/probe_b_d6_25600_losses_mate_window_root_audit_d20d20.json`
  - sampled `24` roots, found `2` bad roots and `1` catastrophic root.
  - `missing_candidate = 0`; both bad roots are ranking/value-horizon issues.
  - worst case: opening `val_shard_00070_2381`, ply 39, arena move
    `h2h4`.  FullPika d20 child eval marks it as `-20000` with mate in child,
    while teacher-best `h8h7` is about `+271cp`.
- exact position replay of that ply with the arena seed shows MCTS can select
  `h2h4` with visit probability about `0.937` and Q about `+0.525`, despite
  the d20 child eval being a mate loss.  This is a self-confirming
  value/horizon basin, not a missing-candidate problem.
- the existing symbolic forcing-check guard catches the position only when
  extended to `5` plies:
  - `3` plies: no event, keeps `h2h4`;
  - `5` plies: replaces `h2h4 -> d9d8`;
  - `7` plies: also replaces `h2h4 -> d9d8`.
- single-opening arena reruns with `--our-root-forcing-check-guard-plies 5`
  both won the opening, but no guard event was recorded because the isolated
  replay diverged earlier than the original six-game trajectory.  Even with an
  adjusted seed, first divergence appeared around ply 23.  Treat these as
  evidence that the opening is not intrinsically lost, not as a fully exact
  rescue proof.
- six-opening rerun with the original d6 `25600` setup plus
  `--our-root-forcing-check-guard-plies 5`:
  `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/d6_6games_25600_forcing_guard5/external_arena_20260526_105349.json`
  - result: `2W-3L-1D / 6 = 41.7%`, same score as the no-guard `25600`
    baseline but with different per-opening results.
  - `events = 0`, so the result changes are search/engine trajectory
    nondeterminism, not guard effects.  This run must not be used as evidence
    that the guard improves or worsens full-game play.
- exact bad-root FEN isolation for opening `val_shard_00070_2381`, game 1
  ply 39:
  - root FEN:
    `3k1ab2/4a2c1/2nNb4/p2R4p/6p2/4C4/P1P3r1P/N3B2r1/4A4/2BK1A3 b`
  - baseline from this root, seed `2026064017`:
    `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/exactroot_g1_ply39_25600_baseline/external_arena_20260526_110235.json`
    - first move `h2h4`, result `opp_win` in `12` plies.
  - same root with `--our-root-forcing-check-guard-plies 5`:
    `/home/laure/alphaxiang/v128_fullpika_root_retune/arena_probe_b/exactroot_g1_ply39_25600_forcing_guard5/external_arena_20260526_110740.json`
    - event: `h2h4 -> h8h7`;
    - reason: `selected_move_allows_opponent_forcing_check_win_5ply`;
    - result: `our_win` in `47` plies.

Interpretation: the user hypothesis is mostly right in shape: by the time the
visible mate blunder appears, the game may already be in a fragile high-risk
basin caused by earlier root-Q/value drift.  But the d20 child audit says the
position itself is not yet hopeless: safer candidate moves are present, and a
5-ply forcing-check guard can reject the catastrophic move at the exact root.
So the failure is better described as "earlier drift into a brittle position
plus a late value/horizon self-confirming blunder", not "already forced loss
where no move matters".

Follow-up implication: do not ship the guard globally from the six-game run.
The right next test is a root-level guard recall/regression suite over d20
catastrophic roots and matched clean controls.  If the symbolic guard catches
catastrophic roots with near-zero clean false positives, it can become a narrow
mate/horizon veto.  If not, keep it as diagnostic signal for future d20
root-regret/value-error training.

## Runner

The runner is:

- `tools/_run_v128_fullpika_root_retune.sh`

It is phase-based. The default phase is `static`, not long training.

Important phases:

- `static`: compile/check scripts only.
- `smoke_existing`: reuse the existing V12 PEAK root-regret JSONL as a cheap
  read-only pipeline smoke, without running a new arena/audit collection or
  training.
- `collect_d5`: collect V12 PEAK vs Pika d5 arena games.
- `collect_d6`: collect V12 PEAK vs Pika d6 arena games.
- `audit`: turn collected arenas into root-decision audits.
- `export`: export root-regret JSONL and trainer shard.
- `read_smoke`: verify shard losses.
- `train_smoke`: run a tiny GPU training smoke.
- `train_probe`: start the first bounded V12.8 retune probe.

The long phases must be launched deliberately.

Recommended next command:

```bash
V128_PHASE=audit \
V128_DEVICE=cuda:0 \
V128_AUDIT_MAX_POSITIONS=120 \
V128_AUDIT_MAX_POSITIONS_PER_FILE=60 \
V128_PIKA_ROOT_DEPTH=20 \
V128_PIKA_CHILD_DEPTH=20 \
V128_PIKA_WORKERS=16 \
V128_PIKA_THREADS_PER_WORKER=2 \
V128_PIKA_HASH_MB=256 \
bash tools/_run_v128_fullpika_root_retune.sh
```

Use the 120-position audit as the first real V12.8 labeling batch only at these
FullPika depths.  The earlier `root d10 / child d12` and `root d14 / child d20`
runs are provisional and must not enter training.

The runner now refuses `export`, `train_smoke`, and `train_probe` unless the
audit metadata satisfies the hard-coded gate:

- `pika_root_depth >= 20`
- `pika_child_depth >= 20`

This gate is not configurable downward by environment variable.
