# PROJECT BRIEF: v13 run_022d Overnight Results

## Summary
- `run_022d` completed normally at `global_step=20000`; no NaNs, no crash, no interrupted state.
- The strongest offline-validation candidates are not the final checkpoint by default. Based on high-sims arena evidence, the current leading candidate is `step18000`.
- `step18000` appears clearly stronger than v12.6-micro on high-search public anchors and has now passed an initial non-Pika Fairy d3 anchor, but still needs CNN-best and larger/fixed-opening panels before being called a shipped checkpoint.

## Training Run

Output directory:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500`

Started from:

`/home/laure/alphaxiang/training_runs/run_022c_v13_nopool_widened_mild_teacherq_from022a1000/snapshots/latest_step1500.pt`

Final checkpoint:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step20000.pt`

Final status:

- `training completed normally at global_step=20000`
- `interrupted=false`
- selfplay buffer samples: `67318`

## Human Validation Checkpoints

Lower is better. Best observed checkpoints:

| Step | human_val_total_loss | Note |
|---:|---:|---|
| 3000 | 3.0555 | offline best / early candidate |
| 10000 | 3.0572 | first-tier candidate |
| 12000 | 3.0615 | first-tier candidate |
| 13000 | 3.0622 | first-tier candidate and current arena leader |
| 18000 | 3.0600 | first-tier candidate |
| 19000 | 3.0619 | first-tier candidate |
| 20000 | 3.0669 | stable but not preferred over earlier candidates |

## High-Sims Candidate Screen

Pika d3 @ `8000` sims, 10 games each:

| Checkpoint | Result | Score | JSON |
|---|---:|---:|---|
| step10000 | 6W-3L-1D | 65.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step10000/pika_d3/external_arena_20260506_015309.json` |
| step12000 | 7W-1L-2D | 80.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step12000/pika_d3/external_arena_20260506_022155.json` |
| step13000 | 7W-0L-3D | 85.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step13000/pika_d3/external_arena_20260506_025415.json` |
| step18000 | 7W-1L-2D | 80.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step18000/pika_d3/external_arena_20260506_032452.json` |
| step19000 | 5W-1L-4D | 70.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step19000/pika_d3/external_arena_20260506_035549.json` |

Pika d4 @ `6400` sims:

| Checkpoint | Games | Result | Score | JSON |
|---|---:|---:|---:|---|
| step13000 | 20 | 9W-5L-6D | 60.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step13000_d4_sims6400_20g/pika_d4/external_arena_20260506_043527.json` |
| step12000 | 10 | 1W-3L-6D | 40.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d4_sims6400_10g/step12000/pika_d4/external_arena_20260506_050513.json` |
| step18000 | 10 | 5W-2L-3D | 65.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d4_sims6400_10g/step18000/pika_d4/external_arena_20260506_052725.json` |
| step18000 extra | 40 | 19W-12L-9D | 58.8% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_d4_sims6400_extra40g/pika_d4/external_arena_20260506_114219.json` |
| step18000 combined | 50 | 24W-14L-12D | 60.0% | combined from the two step18000 d4 JSON files above |

Step18000 d4 combined side balance:

- side games: `25` red / `25` black
- wins by side: `20` red / `4` black
- avg plies: `97.08`

Interpretation: step18000 holds d4 strongly, but the win distribution is red-heavy. This may indicate an opening/first-move pattern or a side-specific conversion gap. It is still a strong anchor result, but should be checked with alternative openings or a larger panel before being treated as pure general strength.

Pika d5 @ `8000` sims:

| Checkpoint | Games | Result | Score | JSON |
|---|---:|---:|---:|---|
| step13000 | 20 | 5W-8L-7D | 42.5% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step13000_d5_sims8000_20g/pika_d5/external_arena_20260506_074559.json` |
| step18000 | 20 | 8W-9L-3D | 47.5% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_d5_sims8000_20g/pika_d5/external_arena_20260506_084121.json` |
| step18000 extra | 30 | 8W-11L-11D | 45.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_d5_sims8000_extra30g/pika_d5/external_arena_20260506_102340.json` |
| step18000 combined | 50 | 16W-20L-14D | 46.0% | combined from the two step18000 JSON files above |

Step18000 combined side balance:

- side games: `25` red / `25` black
- wins by side: `8` red / `8` black
- avg plies: `122.92`

Interpretation: step18000 is currently the best deep-anchor candidate. A 46.0% score over 50 games against Pika d5 @8000 sims is far above the v12.6-micro d5 reference and suggests v13 has genuine deeper-search strength, not only shallow-anchor exploitation.

Pika d3 @ `8000` sims:

| Checkpoint | Games | Result | Score | JSON |
|---|---:|---:|---:|---|
| step18000 | 10 | 7W-1L-2D | 80.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_candidates_d3_sims8000_10g/step18000/pika_d3/external_arena_20260506_032452.json` |
| step18000 extra | 40 | 15W-9L-16D | 57.5% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_d3_sims8000_extra40g/pika_d3/external_arena_20260506_221915.json` |
| step18000 combined | 50 | 22W-10L-18D | 62.0% | combined from the two step18000 d3 JSON files above |

Step18000 d3 combined side balance:

- side games: `25` red / `25` black
- wins by side: `11` red / `11` black
- avg plies: `124.64`

Interpretation: the initial 10-game d3 screen overestimated strength; the 50-game result is a more realistic `62.0%`. This still clearly exceeds v12.6-micro's Pika d3 reference, but it is not an 80% saturated anchor.

## Comparison To v12.6-Micro

v12.6-micro verified reference:

- Pika d3: `18W-21L-11D`, `47.0%`
- Pika d4: `15W-26L-9D`, `39.0%`
- estimated public-anchor strength: `~2080-2150 Elo-equivalent`

Current v13 `step18000` provisional estimate:

- Conservative: `~2300+ Elo-equivalent`
- If non-Pika anchors and larger/fixed-opening d4/d5 panels hold: `~2300-2400 Elo-equivalent`

Important caveat: v13 high-sims results are not directly comparable to v12.6 @1600 sims as a pure model-quality measurement. They do support the separate hypothesis that v13 needs a larger MCTS budget to express its strength.

## Current Judgement
- `step13000` remains a useful comparison point because it led the small d3 screen and had a solid d4 score.
- `step18000` is now the best overall candidate because it scored `62.0%` over 50 games against Pika d3 @8000 sims, `60.0%` over 50 games against Pika d4 @6400 sims, and `46.0%` over 50 games against Pika d5 @8000 sims.
- `step12000` is weaker on d4 in the small screen and is currently deprioritized.

## Non-Pika Anchor

Fairy-Stockfish d3 @ `1600` sims:

| Checkpoint | Games | Result | Score | JSON |
|---|---:|---:|---:|---|
| step18000 | 50 | 46W-2L-2D | 94.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_fairy_d3_sims1600_50g/fairy_d3/external_arena_20260506_224741.json` |

CNN-best @ `1600` sims:

| Checkpoint | Games | Result | Score | JSON |
|---|---:|---:|---:|---|
| step18000 | 50 | 45W-3L-2D | 92.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022d_step18000_cnn_best_sims1600_50g/cnn_best/tournament_20260506_231040.json` |

Interpretation:

- The basic non-Pika generalization check passes.
- Fairy d3 clears the previous `>=90%` ship line.
- CNN-best is below the v12.6-micro reference (`49W-0L-1D`, `99.0%`) but still strong at `92.0%`; this is not a collapse, but it is worth tracking during later d5/full-Pika curriculum.

## Recommended Next Tests
1. Re-test d4 with a different opening seed or fixed opening suite to verify the red-heavy win distribution.
2. Build a Pika d5 failure slice from the `step18000` 50-game d5 panel.
3. Try a short training arm with d5 failure-slice curriculum; do not select by human_val loss alone.
4. Re-check CNN-best after any d5/full-Pika curriculum because the current `92.0%` result leaves less regression margin than v12.6-micro.
5. Use the dedicated failure-analysis brief before launching the next curriculum arm:
   `PROJECT_BRIEF_v13_failure_analysis.md`

## Continuation Follow-Up: run_022e and run_022f

Two continuation probes were launched from the strong `022d` step18000 candidate.

### run_022e: LR 2e-6

Output:

`/home/laure/alphaxiang/training_runs/run_022e_v13_nopool_widened_cont_from022d18000`

Judgement:

- stopped at `global_step=26201` after repeated human-val degradation;
- best observed point was `step20000`, human_val_total_loss `3.0646`;
- later checkpoints were consistently worse (`3.08+` band);
- not prioritized for arena.

### run_022f: LR 5e-7

Output:

`/home/laure/alphaxiang/training_runs/run_022f_v13_nopool_widened_lowlr_from022d18000`

Judgement:

- stopped at `global_step=35106`;
- offline best: `step30000`, human_val_total_loss `3.0558`;
- however, Pika d5 @8000 arena was poor:
  - `step30000`: `1W-8L-11D`, `32.5%`
  - JSON: `/home/laure/alphaxiang/v13_snapshot_smoke/run_022f_step30000_d5_sims8000_20g/pika_d5/external_arena_20260506_201410.json`

Interpretation:

- human validation loss can improve while deep-anchor playing strength regresses;
- this is a concrete warning against selecting v13 checkpoints by offline loss alone;
- `022f` is useful as evidence, but not a ship candidate.

Updated main candidate:

- `022d step18000` remains the leading v13 checkpoint candidate.

## D5 Curriculum Probe: run_023b

Purpose:

- Test whether a high-weight d5 failure-slice curriculum can improve the `022d step18000` deep-anchor weakness.

New data:

- Raw slice: `/home/laure/alphaxiang/v13_d5_curriculum_data/step18000_d5_losses_raw`
- Final labeled slice: `/home/laure/alphaxiang/v13_d5_curriculum_data/step18000_d5_losses_teacherq_d12_all`
- Extraction: `20` Pika d5 loss games, our-turn positions only, `943` samples.
- Labeling:
  - oracle value: Pika depth `15`, `943/943`, errors `0`
  - oracle policy: Pika depth `8`, MultiPV `5`, adaptive temperature, legal smoothing `0.05`, `943/943`, errors `0`
  - teacher_q: Pika depth `12`, `943/943` rows, `6610` entries
- Hygiene audit:
  - `/home/laure/alphaxiang/v13_d5_curriculum_data/step18000_d5_losses_teacherq_d12_all/hygiene_audit.json`
  - dirty shards `0`, oracle illegal entries `0`, teacher_q illegal entries `0`

Training:

- Output: `/home/laure/alphaxiang/training_runs/run_023b_v13_d5_curriculum_warmup0_from022d18000`
- Resume: `022d step18000`
- Reset optimizer/scheduler, `warmup_steps=0`
- New d5 slice sampling ratio: `0.20`
- `teacher_q_loss_weight=0.08`
- Stopped intentionally after `step20000` snapshots were saved because offline validation was degrading and d5 smoke was poor.

Observed offline validation:

| Step | human_val_total_loss | Note |
|---:|---:|---|
| 19000 | 3.0945 | worse than 022d step18000 |
| 20000 | 3.1023 | further degradation |

Arena smoke:

| Checkpoint | Opponent | Games | Result | Score | JSON |
|---|---|---:|---:|---:|---|
| run_023b step19000 | Pika d5 @8000 | 10 | 2W-4L-4D | 40.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023b_d5_sims8000_10g/step19000/pika_d5/external_arena_20260507_003432.json` |
| run_023b step20000 | Pika d5 @8000 | partial | 0W-4L-0D live before stop | 0.0% live | no final JSON; stopped to save compute |

Interpretation:

- The labeled d5 data is clean, but the first curriculum recipe is too aggressive.
- A `20%` loss-slice ratio plus `teacher_q_loss_weight=0.08` appears to damage the already-good `022d step18000` search behavior rather than repair d5.
- Do not ship `run_023b`.
- Keep `022d step18000` as the main candidate.
- If trying curriculum again, use a much softer recipe: lower d5 ratio (`0.03-0.08`), lower teacher_q weight (`0.02-0.04`), and/or train only 300-800 steps before smoke.

## Soft D5 Curriculum Probe: run_023c

Purpose:

- Test whether a much softer d5 curriculum can avoid the `run_023b` collapse while still nudging d5 upward.

Training:

- Output: `/home/laure/alphaxiang/training_runs/run_023c_v13_soft_d5_curriculum_from022d18000`
- Resume: `022d step18000`
- Reset optimizer/scheduler, `warmup_steps=0`
- New d5 slice sampling ratio: `0.05`
- `teacher_q_loss_weight=0.03`
- `policy_oracle_alpha=0.01`
- LR `2e-6`
- Ran only `800` steps, completed normally at `step18800`.

Observed offline validation:

| Step | human_val_total_loss | Note |
|---:|---:|---|
| 18400 | 3.0996 | worse than 022d step18000 |
| 18800 | 3.0995 | still worse; not selected by offline loss |

Arena smoke:

| Checkpoint | Opponent | Games | Result | Score | JSON |
|---|---|---:|---:|---:|---|
| run_023c step18800 | Pika d5 @8000 | 10 | 4W-4L-2D | 50.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023c_step18800_d5_sims8000_10g/pika_d5/external_arena_20260507_013604.json` |
| run_023c step18800 | Pika d5 @8000 extra | 10 | 2W-3L-5D | 45.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023c_step18800_d5_sims8000_extra10g/pika_d5/external_arena_20260507_032705.json` |
| run_023c step18800 | Pika d5 @8000 combined | 20 | 6W-7L-7D | 47.5% | combined from the two run_023c d5 JSON files above |
| run_023c step18800 | Pika d4 @6400 | 10 | 4W-2L-4D | 60.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023c_step18800_pika_d4_10g/pika_d4/external_arena_20260507_020419.json` |
| run_023c step18800 | Pika d3 @8000 | 10 | 4W-4L-2D | 50.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023c_step18800_pika_d3_10g/pika_d3/external_arena_20260507_023103.json` |
| run_023c step18800 | Pika d3 @8000 extra | 10 | 6W-2L-2D | 70.0% | `/home/laure/alphaxiang/v13_snapshot_smoke/run_023c_step18800_pika_d3_extra10g/pika_d3/external_arena_20260507_030431.json` |
| run_023c step18800 | Pika d3 @8000 combined | 20 | 10W-6L-4D | 60.0% | combined from the two run_023c d3 JSON files above |

Interpretation:

- Softer d5 curriculum avoids the immediate collapse seen in `run_023b`.
- d5 smoke is neutral/slightly promising but not a clear improvement over `022d step18000` (`46.0%` over 50 games); combined `run_023c` d5 is `47.5%` over 20 games.
- d4 appears preserved in the small smoke.
- d3 appears roughly preserved after an extra 10-game check; combined `run_023c` d3 is `60.0%` over 20 games versus the `022d step18000` 50-game reference of `62.0%`.
- Do not promote `run_023c` as a new main checkpoint yet; the current evidence says it is probably comparable to `022d step18000`, not clearly stronger.
- Main candidate remains `022d step18000` until a larger d5 panel proves `run_023c` is meaningfully better.
