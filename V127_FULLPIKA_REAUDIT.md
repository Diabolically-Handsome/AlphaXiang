# V12.7 FullPika Re-Audit

Date: 2026-05-24

## Summary

The V12.7 FullPika Curriculum run did invoke stronger Pikafish labeling, but it
was not a clean test of "full-strength Pika drives V12 training." The run is
better interpreted as a light continuation from V12 PEAK with a small amount of
new FullPika-labeled data mixed into a large protected training distribution.

This means the weak final d5/d6 result should not be treated as proof that the
V12 route is exhausted.

## Evidence

Run:

- train: `/home/laure/alphaxiang/training_runs/run_020_v127_fullpika_curriculum`
- data: `/home/laure/alphaxiang/selfplay_runs_v127_fullpika_curriculum`
- eval: `/home/laure/alphaxiang/v127_fullpika_curriculum_eval`

Final checkpoint:

- `/home/laure/alphaxiang/training_runs/run_020_v127_fullpika_curriculum/snapshots/latest_step298000.pt`

Final ladder:

| opponent | result | score |
|---|---:|---:|
| Pika d3 | 21-20-9 / 50 | 51.0% |
| Pika d4 | 12-31-7 / 50 | 31.0% |
| Pika d5 | 6-41-3 / 50 | 15.0% |
| Pika d6 | 1-44-5 / 50 | 7.0% |

The teacher settings were stronger than the original V12 teacher:

- distill depth: 20
- oracle value depth: 20
- oracle policy depth: 14
- oracle policy MultiPV: 5
- policy oracle alpha: 0.5
- distill hash: 256 MB
- distill workers: 12

But the actual training signal was much weaker than "FullPika takeover":

- trainer ran with ordinary policy/value/WDL losses enabled;
- `teacher_q_loss_weight=0.0`;
- `teacher_q_pairwise_loss_weight=0.0`;
- `bad_move_suppression_loss_weight=0.0`;
- no action-value/teacher-Q labels were used;
- train logs show about `mix=0.90/0.10`, so only about 10% of each batch came
  from the new selfplay/distill buffer;
- the final buffer had only `34,468` new samples;
- each cycle trained before the newly generated shards were oracle-labeled, so
  FullPika labels affected later cycles with delay, and the last d6 cycle's
  fresh labels were not consumed by any later training cycle.

There was also one aborted d5 oracle labeling attempt before the fixed restart:

- `stage1_c001_20260518_082305_label_d5n05.log`
- recorded in `stage1_driver.log` with `rc=1`

The fixed restart later completed d5 and d6 labels successfully, but this
history reinforces that the first V12.7 attempt was not a pristine teacher
experiment.

## Interpretation

The run did not fail because Pikafish d20/d14 was never used. It did use it.

The more precise failure mode is:

1. FullPika supervision was low-frequency relative to the old/human-protected
   distribution.
2. The training objective remained ordinary policy/value/WDL continuation, not
   root-regret, teacher-Q, or pairwise candidate ranking.
3. The freshest hard d5/d6 data arrived too late in each cycle to shape that
   same cycle's training.
4. The final d6 labels were generated after the final training block and were
   effectively audit artifacts, not learned training signal.

So V12.7 did not actually answer the user's intended question:

> If a V12-sized model is trained with truly strong Pika supervision and a sane
> curriculum, can it move past the d5/d6 bottleneck without V13-scale cost?

## Recommendation

Do not resume the old V12.7 run as-is.

The next useful V12 experiment should be diagnostic-first and cheaper:

1. Freeze the existing V12 PEAK and V12.7 step298000 checkpoints.
2. Run a paired root audit on V12 PEAK vs V12.7 for d4/d5 losses.
3. Measure whether V12.7's failures are missing-candidate, ranking/Q inversion,
   or horizon/mate, using the same root decision audit style that worked for
   V13.
4. If V12.7 shows a cleaner/fixable failure profile, build a small fully-labeled
   root-regret dataset before any further training.
5. If training resumes, enforce "generate -> full oracle label -> train" order,
   raise the new-data sampling ratio explicitly, and use teacher-Q/pairwise only
   in a narrow root-regret setup.

This keeps V12.7 alive as a cheaper research platform without pretending the
previous FullPika curriculum was a decisive negative result.

## 6400-Sim Root Recheck

The first V12.7 d4/d5 root-audit smoke used `1600` sims because that matched
the old V12.7 ladder script. That is too low for a serious V12 strength
judgment, so I treated it only as a cheap prefilter.

Prefilter:

- input: final V12.7 d4/d5 black-side loss roots
- search: V12.7 step298000, `1600` sims
- teacher: Pika root d8 / child d8
- output: `v127_reaudit/v127_step298000_d4d5_black_loss_root_audit_d8.json`

The prefilter found 8 suspicious bad roots. I then reran exactly those roots
with `6400` sims and classified the move selected by the rerun MCTS, not the
old arena move:

- V12.7 output:
  `v127_reaudit/v127_step298000_bad_roots_selected_mcts6400_d8.json`
- V12 PEAK output:
  `v127_reaudit/v12_peak_bad_roots_selected_mcts6400_d8.json`

Results on those 8 suspicious roots:

| checkpoint | bad roots | catastrophic | missing-candidate | q-inversion |
|---|---:|---:|---:|---:|
| V12 PEAK step286000 | 2 / 8 | 2 / 8 | 0 / 8 | 0 / 8 |
| V12.7 step298000 | 3 / 8 | 1 / 8 | 0 / 8 | 2 / 8 |

Per-root read:

- V12.7 fixed one V12 PEAK catastrophic root: `d9b9` regret `19237cp` became
  `g6h6` regret `57cp`.
- V12.7 kept one catastrophic root unchanged: `g3e3` regret `19121cp`.
- V12.7 introduced two moderate ranking/Q failures on d4 roots:
  - `i6i5`, regret `244cp`
  - `d9e9`, regret `312cp`

Interpretation:

- Higher sims do help: the original 8 record-selected bad roots shrink to 3
  actual MCTS-selected bad roots at `6400` sims.
- But V12.7 still has real high-sim root ranking failures.
- The dominant issue remains ranking/value/search calibration, not missing
  candidates: missing-candidate is still `0 / 8`.
- V12.7 is not a clean upgrade over V12 PEAK on this tiny suspicious-root
  slice. It fixed one catastrophic decision, but introduced smaller Q/ranking
  errors elsewhere.

This supports a cautious V12 continuation only if the next experiment is
fully-labeled and root-regret focused, with explicit regression guards against
d4/d5 root-ranking damage.

## 6400-Sim Arena Smoke

To check whether the old `1600`-sim ladder was simply under-searching V12.7, I
ran a tiny Pika d5 arena smoke at `6400` sims:

- checkpoint:
  `/home/laure/alphaxiang/training_runs/run_020_v127_fullpika_curriculum/snapshots/latest_step298000.pt`
- output:
  `v127_reaudit/arena_6400_v127_d5_smoke/external_arena_20260524_120143.json`
- config: Pika d5, 4 games, parallel 4, `c_puct=1.25`, `q_weight=1.0`,
  `temp=0.1`, no tactical guards

Result:

- `0W-3L-1D / 4 = 12.5%`
- mate losses: 3
- no-capture draw: 1

This tiny sample is not a final estimate, but it does not support the idea that
`6400` sims alone unlocks V12.7 d5 strength.

I also started a same-seed `6400` + mate1/mate2 guard smoke, but stopped it
after roughly 9 minutes without a completed game. The process was active rather
than crashed, but the cost/benefit was poor:

- command included root mate1/mate2 guard and tactical mate1/mate2 extension;
- CPU was active at about 11 cores;
- no game result was produced before stopping.

If guard evaluation is revisited, it should use a narrower configuration first:
root guard only, 1-2 serial games, or root-only offline replay before arena.

## Expanded 80-Root High-Sim Audit

After the initial 8-root recheck, I reran the full d4/d5 black-side loss-root
sample with the corrected selected-move semantics:

- selected move: rerun MCTS best move (`--selected-source mcts`)
- search: `6400` sims, `c_puct=1.25`, `q_weight=1.0`, `temp=0.1`
- teacher: Pika root d8 MultiPV6 + child d8
- positions: 80 roots, 40 from final d4 black losses and 40 from final d5
  black losses

Outputs:

- V12.7:
  `v127_reaudit/v127_step298000_d4d5_black_loss_selected_mcts6400_d8_full80.json`
- V12 PEAK:
  `v127_reaudit/v12_peak_d4d5_black_loss_selected_mcts6400_d8_full80.json`

Summary:

| checkpoint | bad roots | catastrophic | missing-candidate | q-inversion | root verifier detectable |
|---|---:|---:|---:|---:|---:|
| V12 PEAK step286000 | 6 / 80 | 1 / 80 | 0 / 80 | 5 / 80 | 0 / 80 |
| V12.7 step298000 | 5 / 80 | 1 / 80 | 0 / 80 | 3 / 80 | 1 / 80 |

V12.7 did learn something, but the net effect is small:

- fixed by V12.7: 5 roots
- newly bad in V12.7: 4 roots
- bad in both: 1 root
- net: only 1 fewer bad root on this 80-root slice

Notable examples:

| root | V12 PEAK | V12.7 | interpretation |
|---|---:|---:|---|
| d5 game 11 ply 33 | `f7f1`, regret `312cp` | `g0g5`, regret `0cp` | fixed |
| d5 game 3 ply 57 | `c7c1`, regret `247cp` | `f1h1`, regret `22cp` | mostly fixed |
| d4 game 3 ply 193 | `e6d6`, regret `0cp` | `d9e9`, regret `312cp` | new ranking regression |
| d5 game 5 ply 33 | `a9b9`, regret `0cp` | `d9e9`, regret `323cp` | new ranking regression |
| d5 game 5 ply 65 | catastrophic in both, regret `19121cp` | catastrophic in both, regret `19121cp` | unfixed |

Interpretation:

- The original V12.7 FullPika run did not simply fail to teach the model. It
  moved root decisions in both directions.
- The failure pattern is the same one seen in V13: good moves are usually
  already present; the system sometimes ranks them incorrectly.
- Continuing from V12.7 step298000 is risky because it already contains some
  new d4/d5 regressions.
- A cleaner next experiment should restart from V12 PEAK, use fully labeled
  root-regret data, and gate every snapshot against the 80-root high-sim audit
  before any arena.

## Root-Regret Data Export

I exported the V12 PEAK high-sim audit into a trainer-readable root-regret
dataset:

- all-candidate JSONL:
  `v127_reaudit/v12_peak_root_regret_d4d5_black_6400_d8_all.jsonl`
- selected/refuted JSONL:
  `v127_reaudit/v12_peak_root_regret_d4d5_black_6400_d8_selected_or_refuted.jsonl`
- tensor shard:
  `v127_reaudit/v12_peak_root_regret_shard_6400_d8/train/shard_00000.pt`

Export stats:

- roots: 80
- all-candidate rows: 1610
- selected/refuted rows: 813
- train shard samples: 63
- train bad roots: 6
- train catastrophic roots: 1

Shard read smoke passed on CPU:

- has teacher-Q: yes
- has legal mask: yes
- has bad-move labels: yes
- teacher-Q pairwise loss: `0.9546`
- bad-move suppression loss: `2.1358`
- valid pairwise samples in smoke batch: 4
- valid bad-move suppression samples in smoke batch: 4

This showed that the data plumbing for a V12 root-regret micro-experiment was
ready before launching any training smoke.

## Offline Policy Comparison

I also ran offline policy-ranking evals on the exported root-regret JSONLs.

Self-eval:

| checkpoint | JSONL | bad roots | catastrophic | bad safe-top repairs | non-bad regressions | mean top-candidate regret |
|---|---|---:|---:|---:|---:|---:|
| V12 PEAK | V12 PEAK audit | 6 | 1 | 3 | 7 / 72 | 1028.8cp |
| V12.7 | V12.7 audit | 5 | 1 | 2 | 3 / 73 | 533.3cp |

Cross-eval:

| checkpoint | JSONL | bad safe-top repairs | catastrophic repairs | new non-bad regressions vs anchor | mean top-candidate regret |
|---|---|---:|---:|---:|---:|
| V12.7 | V12 PEAK audit | 5 | 1 | 1 / 72 | 533.3cp |
| V12 PEAK | V12.7 audit | 1 | 0 | 4 / 73 | 1028.8cp |

Interpretation:

- V12.7 did improve policy ranking on this root-candidate distribution.
- The improvement did not reliably convert into better 6400-sim MCTS root
  choices; V12.7 only reduced bad roots from 6/80 to 5/80.
- The remaining bottleneck is therefore not simply "policy never saw the good
  move." It is the policy/value/search interaction at the root.
- A pure policy-head root-regret finetune is unlikely to be enough by itself.
  If trained at all, it needs an offline root-MCTS recheck gate, not just policy
  top-candidate metrics.

## V12 Root-Regret Micro Runner

I added a conservative runner for the next V12 experiment:

- script:
  `tools/_run_v12_root_regret_micro.sh`
- base checkpoint:
  `/home/laure/alphaxiang/PEAK_step286000_v12_probe2_score95pct_d1.pt`
- default data:
  `v127_reaudit/v12_peak_root_regret_shard_6400_d8`
- default formal output:
  `/home/laure/alphaxiang/training_runs/run_021_v12_root_regret_micro_from_peak`

The runner does not resume old V12.7. It restarts from V12 PEAK and only trains
the policy-head projections:

- trainable params: `263,682 / 38,610,182`
- ordinary policy/value/WDL losses: off
- teacher-Q CE: off
- teacher-Q pairwise: on
- bad-move suppression: on
- anchor policy KL to V12 PEAK: on
- value anchor: off

This is deliberately a tiny root-ranking probe, not a new general training run.

Static and data-entry checks passed:

- `bash -n tools/_run_v12_root_regret_micro.sh`
- `py_compile` for the audit/eval/smoke tools
- shard read smoke confirms teacher-Q, legal masks, and bad-move labels are
  present.

GPU training smoke also passed after using `micro_batch=64`:

- smoke checkpoint:
  `/tmp/v12_root_regret_micro_smoke_train_gpu4/latest.pt`
- step 1 was a harmless human fallback before the selfplay shard passed the
  trainer's stability check.
- step 2 ingested the root-regret shard:
  - `added_shards=1`
  - `selfplay_buffer=63`
  - `mix=0.00/1.00`
  - `teacher_q_pairwise=2.6504`
  - `bad_move_suppression=2.2112`
  - `n_pairwise=9`
  - `n_bad_suppress=9`

The 2-step smoke checkpoint evaluates cleanly:

- output:
  `v127_reaudit/v12_root_regret_micro_smoke_policy_eval.json`
- checkpoint step: `286002`
- new non-bad regressions vs anchor: `0 / 72`
- median anchor KL: `0.00000246`

This smoke is not evidence of playing strength. Its value is narrower but
important: the V12 root-regret training path is now real and verified.

The runner now includes a required post-training MCTS gate:

- phase: `mcts_gate_base`
- phase: `mcts_gate_a`
- phase: `mcts_gate_compare`

The gate reruns the same 80-root audit with:

- selected move: rerun MCTS best move
- sims: `6400`
- teacher: Pika root d8 / child d8
- source audit:
  `v127_reaudit/v12_peak_d4d5_black_loss_selected_mcts6400_d8_full80.json`

Acceptance is intentionally strict before any arena:

- candidate bad roots must be lower than the V12 PEAK baseline;
- catastrophic roots must not increase;
- missing-candidate count must not increase.

This directly addresses the `1600 sims` concern: all root-level acceptance for
V12 now uses `6400 sims`.

## V12 Policy-Head Micro Probe Results

I ran several tiny policy-head-only root-regret probes from V12 PEAK. These were
diagnostic probes, not release candidates:

| run | lr | anchor KL | steps | bad safe-top repairs | catastrophic repairs | new non-bad regressions |
|---|---:|---:|---:|---:|---:|---:|
| `run_021_v12_root_regret_micro_from_peak` | `2e-6` | `0.05` | 250 | 6 / 6 | 1 / 1 | 3 / 72 |
| `run_021b_v12_root_regret_micro100_from_peak` | `2e-6` | `0.05` | 100 | 6 / 6 | 1 / 1 | 3 / 72 |
| `run_021c_v12_root_regret_micro25_from_peak` | `2e-6` | `0.05` | 25 | 4 / 6 | 1 / 1 | 3 / 72 |
| `run_021d_v12_root_regret_micro100_lr2e7_anchor20_from_peak` | `2e-7` | `0.20` | 100 | 6 / 6 | 1 / 1 | 3 / 72 |

The policy-level result is real: the root-regret signal can move V12's policy
toward the teacher-best candidate on these bad roots. But the side effect is
also real: even shallow/low-LR policy-head changes introduce `3 / 72` new
non-bad regressions on this small root set.

The best-behaved probe was:

- `/home/laure/alphaxiang/training_runs/run_021d_v12_root_regret_micro100_lr2e7_anchor20_from_peak/arm_a_policy_head/latest.pt`

Offline policy eval:

- bad safe-top repairs: `6 / 6`
- catastrophic repairs: `1 / 1`
- new non-bad regressions vs anchor: `3 / 72`
- mean top-candidate regret: `27.7cp`
- median top-candidate regret: `0.5cp`

I then ran the required `6400`-sim MCTS gate:

- output:
  `/home/laure/alphaxiang/v12_root_regret_micro100_lr2e7_anchor20/mcts_gate/arm_a_policy_head_latest_selected_mcts6400_d8.json`

Comparison against V12 PEAK on the same 80 roots:

| checkpoint | bad roots | catastrophic | missing-candidate | Q inversion | selected mean regret |
|---|---:|---:|---:|---:|---:|
| V12 PEAK step286000 | 6 / 80 | 1 / 80 | 0 / 80 | 5 / 80 | 278.2cp |
| micro100 lr2e-7 anchor0.20 | 6 / 80 | 1 / 80 | 0 / 80 | 4 / 80 | 270.9cp |

MCTS gate verdict:

- pure policy-head root-regret repair did **not** reduce high-sim bad-root
  count;
- it slightly reduced Q inversion and mean regret, but not enough to justify
  arena;
- it made Pika root d8 replacement detectable on 2 roots, suggesting that a
  root-level gate/verifier may be more promising than changing the checkpoint.

Conclusion:

- freeze these policy-head checkpoints as diagnostics;
- do not run arena with them;
- next V12 step should be a conservative root-level gate/verifier over V12
  MCTS top-K, using `6400` sims as the decision surface.

## V12 Root Verifier Probe

I added a generic offline verifier-grid tool:

- `tools/root_topk_verifier_offline_eval.py`

It replays the same basic rule as `external_arena.py --our-pikafish-verifier`
against an existing root-decision audit:

- candidate set: MCTS top-K plus selected;
- verifier label: stored Pika child eval from the audit;
- override rule: child eval improvement from opponent POV >= margin, with an
  optional danger threshold.

On the V12 PEAK 80-root `6400` audit, the offline grid found strong upper-bound
signals:

- V12 PEAK, top16/margin120/danger100:
  - bad roots: `6 -> 0`
  - clean regressions: `0`
  - catastrophic: `1 -> 0`
- V12 PEAK, top6/margin120/danger100:
  - bad roots: `6 -> 1`
  - clean regressions: `0`
  - catastrophic: `1 -> 0`
- micro100 lr2e-7 anchor0.20, top6/margin120/danger100:
  - bad roots: `6 -> 0`
  - clean regressions: `0`
  - catastrophic: `1 -> 0`

Outputs:

- `v127_reaudit/v12_peak_topk_verifier_offline_grid.json`
- `v127_reaudit/v12_peak_topk_verifier_offline_grid.md`
- `v127_reaudit/v12_micro100_lr2e7_anchor20_topk_verifier_offline_grid.json`
- `v127_reaudit/v12_micro100_lr2e7_anchor20_topk_verifier_offline_grid.md`

I then ran a tiny paired d5 smoke with V12 PEAK, `6400` sims, same seed and same
two openings:

| config | result | score | events | duration |
|---|---:|---:|---:|---:|
| baseline | 1W-1L-0D | 50.0% | 0 | 285.9s |
| top6 verifier d8 | 1W-1L-0D | 50.0% | 9 | 366.0s |
| top6 verifier d8 + root mate1/mate2 guard | 1W-0L-1D | 75.0% | 8 | 431.7s |

The important event was the second opening:

- baseline: mate loss at 137 plies;
- verifier only: mate loss delayed to 213 plies;
- verifier + root mate1/mate2: converted to no-capture draw at 273 plies.

This is still only a 2-game smoke, but it is the first V12-side result where the
mechanism matches the audit:

- top-K child verifier catches dangerous root choices;
- root mate guard handles the final forced-mate leakage;
- no checkpoint change is needed.

Next gate:

- run a 12-opening d5 paired test before considering d6 or any further V12
  training.
