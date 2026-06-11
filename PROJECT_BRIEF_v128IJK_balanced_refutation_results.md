# v12.8I/J/K Balanced Refutation Results

Date: 2026-05-03

## Goal

Continue the v12.8H full-Pika teacher idea without letting the d5-loss
refutation slice dominate training.

The working hypothesis was:

- v12.8H failed because refutation data was too concentrated.
- If refutation becomes a low-frequency regularizer, we might preserve the
  v12.8E d4 gain while improving d5.

## Code Changes

- `xiangqi_train.py`
  - Added `--selfplay-dir-sampling-ratios`.
    - Self-play shards can now be sampled by configured source ratios instead
      of pooled shard-size probability.
    - This avoids the old failure where a tiny refutation directory could be
      selected as the one sampled shard and fill most/all self-play samples in
      a batch.
  - Added `--policy-loss-weight`.
    - Default remains `1.0`.
    - Allows teacher-Q/anchor-only micro-finetunes with ordinary policy CE
      disabled.
- Added scripts:
  - `tools/_run_v128i_balanced_refutation_train.sh`
  - `tools/_run_v128i_after_balanced_smokes.sh`
  - `tools/_run_v128j_teacherq_only_refutation_train.sh`
  - `tools/_run_v128j_after_teacherq_only_smokes.sh`
- Added diagnostics:
  - `tools/teacher_q_alignment.py`
  - `tools/interpolate_checkpoints.py`

Verification:

- `py_compile` passed for modified Python files.
- `bash -n` passed for new scripts.
- No training or arena process was left running after this round.

## Calibration Baseline

The earlier v12.8E quick result used only 12 d3/d4 games and looked unusually
high. I reran the same checkpoint at 20 games:

- Checkpoint:
  - `/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt`
- Output:
  - `/home/laure/alphaxiang/v128_snapshot_smoke/v128e_anchor_rerun20`

Result:

- Pika d3: `6-9-5`, score `42.5%`
- Pika d4: `10-6-4`, score `60.0%`

Interpretation:

- v12.8E's d4 improvement is real.
- v12.8E's d3 is weaker than the original 12-game quick result suggested.

## v12.8I: Balanced Ordinary Finetune

Run:

- `/home/laure/alphaxiang/training_runs/run_019k_v128i_balanced_refute_anchor_e`

Key params:

- Resume/anchor: v12.8E step297000.
- Refutation data: clean x1 full-Pika d5-loss slice.
- Sampling ratios: stage2 `0.74`, d4 slice `0.20`, refute `0.06`.
- Ordinary losses still enabled.
- `teacher_q_loss_weight=0.08`.
- `policy_oracle_alpha=0.03`.

Results:

- step297250:
  - d3: `8-10-2`, score `45.0%`
  - d4: `1-13-6`, score `20.0%`
- step297500:
  - d3: `6-8-6`, score `45.0%`
  - d4: `2-13-5`, score `22.5%`

Conclusion:

- Even balanced sampling did not solve the problem.
- Ordinary policy/value/WDL loss plus the refutation slice still destroyed the
  d4 style learned by v12.8E.

## v12.8J: Teacher-Q Only

Run:

- `/home/laure/alphaxiang/training_runs/run_019l_v128j_teacherq_only_refute`

Key params:

- Resume/anchor: v12.8E step297000.
- `policy_loss_weight=0.0`
- `value_loss_weight=0.0`
- `wdl_loss_weight=0.0`
- `teacher_q_loss_weight=0.12`
- Anchor KL/value enabled.
- Replay buffer reduced so self-play/refute actually appears in each batch.

Smoke:

- step297250 d3: `6-12-2`, score `35.0%`
- step297250 d4 partial before stopping: `1-10-5` after 16 games, score `21.9%`

Conclusion:

- Disabling ordinary losses did not rescue the approach.
- Teacher-Q-only still pushed the adapter into a public-anchor failure mode.

## Offline teacher_q Alignment

Data:

- `/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses`

Outputs:

- `/home/laure/alphaxiang/v128_offline_diagnostics/teacher_q_alignment`

Summary:

| checkpoint | teacher_q rows | top1 agreement | teacher top1 in model top3 | median shard CE |
|---|---:|---:|---:|---:|
| v12.8E step297000 | 465 | 38.3% | 67.7% | 2.904 |
| v12.8I step297500 | 465 | 38.9% | 68.0% | 2.871 |
| v12.8J step297250 | 465 | 38.1% | 67.3% | 2.855 |

Interpretation:

- I/J only slightly improved listwise teacher-Q CE.
- They did not materially improve teacher top-1 agreement.
- Yet they caused major d4 regression.
- This suggests the adapter update is not learning a robust tactical prior; it
  is perturbing a fragile search/style balance.

## v12.8K: Weight Interpolation

I interpolated v12.8E with v12.8J to test whether a tiny amount of teacher-Q
update could be kept while preserving d4.

Generated:

- `/home/laure/alphaxiang/training_runs/run_019m_v128k_interp_e_j/snapshots/interp_alpha002.pt`
- `/home/laure/alphaxiang/training_runs/run_019m_v128k_interp_e_j/snapshots/interp_alpha010.pt`
- `/home/laure/alphaxiang/training_runs/run_019m_v128k_interp_e_j/snapshots/interp_alpha020.pt`
- `/home/laure/alphaxiang/training_runs/run_019m_v128k_interp_e_j/snapshots/interp_alpha035.pt`

Smoked:

- alpha `0.10`, 12 games:
  - d3: `7-4-1`, score `62.5%`
  - d4: `2-6-4`, score `33.3%`
- alpha `0.02`, 12 games:
  - d3: `3-8-1`, score `29.2%`
  - d4: `1-6-5`, score `29.2%`

Conclusion:

- Interpolation did not find a safe region.
- The d4 anchor is extremely sensitive to this direction in parameter space.

## Overall Conclusion

The full-Pika refutation dataset is clean and diagnostically valuable, but
using it to finetune the small v12 global-strategy adapter is not currently a
ship path.

The repeated failure pattern is now consistent:

- refutation/teacher-Q signal can move behavior;
- but it moves behavior along a Pareto direction:
  - sometimes d3 improves,
  - d4 collapses,
  - d5 is not fixed robustly.

This is strong evidence that the bottleneck is not simply "teacher too weak".
It looks more like limited adapter/trunk capacity or a fragile representation
that cannot absorb tactical refutation without breaking the d4 style.

## Recommendation

Do not ship v12.8I/J/K.

For v12:

- Keep v12.8E as the most interesting research checkpoint, not a ship candidate.
- Keep the full-Pika refutation data for diagnostics and future curriculum.
- Stop further adapter-only refutation finetunes unless a new mechanism changes
  the optimization geometry.

For v13:

- Carry global strategic tokens forward as a native architecture feature.
- Use full-Pika teacher-Q as a low-frequency auxiliary diagnostic/curriculum,
  not as a dominant small-adapter finetune.
- Prefer more capacity or better integrated trunk adaptation over more
  aggressive refutation weighting.
