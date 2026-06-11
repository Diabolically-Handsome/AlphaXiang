# v13 026 DPO / bad-move repair smoke results

## Summary

- Implemented DPO-style teacher-Q pairwise repair with a frozen anchor reference.
- Added an offline repair drift diagnostic for policy KL, top-k overlap, value drift, and teacher-Q regret.
- Ran 026a/026b/026c/026d micro-smokes from the practical best V13 checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`
- Result: DPO reference strongly reduces global policy drift versus 025h, but the current repair loss is not yet a breakthrough. Do not promote 026 checkpoints.

## Code Changes

- `xiangqi_train.py`
  - Added `--teacher-q-pairwise-beta`.
  - Added `--teacher-q-pairwise-use-anchor-reference`.
  - Added DPO-style pairwise loss:
    `new_good_bad_gap - reference_good_bad_gap - margin`.
  - Added `--teacher-q-pairwise-bad-move-only`.
  - Plumbed optional shard field `bad_move` through extraction, collation, device transfer, and teacher-Q pairwise loss.

- `tools/policy_repair_drift_diagnostic.py`
  - New read-only diagnostic tool comparing a candidate repair checkpoint against a frozen reference.
  - Reports policy KL, top-1 change rate, top-3/top-5 overlap, value drift, teacher-Q regret, and known-bad-move metrics when `bad_move` exists.

## Runs

### 026a

- Run:
  `/home/laure/alphaxiang/training_runs/run_026a_v13_dpo_pairwise_anchor_ref_smoke_from022d18000`
- Issue found:
  - Forgot to shrink `replay_buffer_size`.
  - Batch mix stayed `1.00/0.00`; no repair samples entered the batch.
  - `n_pairwise=0`.
- Verdict:
  - Not a valid repair test, but useful sampler lesson.

### 026b

- Run:
  `/home/laure/alphaxiang/training_runs/run_026b_v13_dpo_pairwise_anchor_ref_smoke_from022d18000`
- Fix:
  - Set `--replay-buffer-size 192`.
  - Batch mix became `0.35/0.65`.
  - `n_pairwise ~= 8`.
- Offline diagnostic on d5 teacher-Q-all:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026b_vs_022d_teacherq_d12_all.json`
  - policy KL mean: `0.375`
  - top1 change: `12.1%`
  - value drift mean: `0.00157`
  - median regret delta: `+1cp`
- Verified blunder diagnostic:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026b_vs_022d_combined100_verified.json`
  - median regret delta: `+17.5cp`
- Verdict:
  - Much safer than 025h, but too weak / not repairing the target slice.

### 026c

- Run:
  `/home/laure/alphaxiang/training_runs/run_026c_v13_dpo_pairwise_stronger_anchor_ref_smoke_from022d18000`
- Change:
  - Stronger DPO pairwise: weight `6.0`, beta `1.0`.
- Offline diagnostic on d5 teacher-Q-all:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026c_vs_022d_teacherq_d12_all.json`
  - policy KL mean: `0.352`
  - top1 change: `11.5%`
  - value drift mean: `0.00161`
  - median regret delta: `+1cp`
- Verified blunder diagnostic:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026c_vs_022d_combined100_verified.json`
  - policy KL mean: `0.372`
  - top1 change: `14.2%`
  - median regret delta: `-8.5cp`
- d5 arena smoke:
  `/home/laure/alphaxiang/v13_refutation_curriculum/smoke_run026c_step18050_d5_sims8000_6g_seed202605823_cuda0/external_arena_20260508_124234.json`
  - `1W-2L-3D`, score `41.7%`
  - red: `1W-1L-1D`, score `50.0%`
  - black: `0W-1L-2D`, score `33.3%`
- Verdict:
  - Best 026 candidate so far.
  - Does not collapse, but not enough evidence to beat 022d.

### 026d

- Run:
  `/home/laure/alphaxiang/training_runs/run_026d_v13_dpo_badmove_anchor_ref_smoke_from022d18000`
- Change:
  - `--teacher-q-pairwise-bad-move-only`.
  - Only compares teacher-best against the recorded `bad_move`.
- Offline diagnostic on verified blunders:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026d_vs_022d_combined100_verified.json`
  - policy KL mean: `0.409`
  - top1 change: `15.5%`
  - median regret delta: `-10cp`
  - known bad rows: `40`
  - known bad median rank stayed `1.0 -> 1.0`
- Offline diagnostic on d5 teacher-Q-all:
  `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_026d_vs_022d_teacherq_d12_all.json`
  - policy KL mean: `0.404`
  - top1 change: `16.8%`
  - median regret delta: `+3.75cp`
- Verdict:
  - More targeted, but still not actually pushing known bad moves down enough.
  - Do not arena-test unless the loss is strengthened or redesigned.

## Comparison Against 025h

025h diagnostic on d5 teacher-Q-all:
`/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_025h_vs_022d_teacherq_d12_all.json`

- policy KL mean: `0.761`
- top1 change: `28.1%`
- value drift mean: `0.0143`
- median regret delta: `+1.25cp`

026b/026c/026d confirm the pasted advice:

- DPO-style reference does reduce global-policy drift.
- Anchor alone is not enough; repair must explicitly lower the known bad move.
- Current DPO margin is still too weak or too indirect to create a ship candidate.

## Recommendation

- Keep the DPO/reference code and the drift diagnostic.
- Do not promote 026b/026c/026d.
- Next experiment should not be another broad finetune.
- Next best technical move:
  - implement a direct bad-move suppression term:
    `loss_bad = -log sigmoid(beta * ((logp_good - logp_bad) - ref_gap - margin))`
    plus an explicit `bad_logit_down` penalty only on the known bad move;
  - or build the frozen-V13 + small tactical repair head/gate, so the base V13 logits remain untouched.

