# v13 Refutation Curriculum 025 Results

## Summary

- No new checkpoint is promoted over `run_022d_v13_nopool_widened_overnight_from022c1500/latest_step18000.pt`.
- The main discovery is diagnostic: v13 can learn the d5 failure slice, but direct teacher-Q imitation damages global playing strength.
- Best current ship recommendation remains v13/022d; keep 025-series work as a research branch for regret repair.

## Ground Truth Baseline

- Baseline checkpoint:
  `/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`
- d5 control, same seed as 025 smokes:
  `/home/laure/alphaxiang/v13_refutation_curriculum/control_run022d_step18000_d5_sims8000_6g_seed202605721_cuda0/pika_d5/external_arena_20260508_031750.json`
  - Pika d5 @8000: `1W-1L-4D`, score `50.0%`
- d4 control, same seed as 025d:
  `/home/laure/alphaxiang/v13_refutation_curriculum/control_run022d_step18000_d4_sims8000_6g_seed202605722_cuda0/pika_d4/external_arena_20260508_051840.json`
  - Pika d4 @8000: `3W-1L-2D`, score `66.7%`

## Data And Diagnostics Built

- Added `tools/arena_loss_drop_scan.py` to scan lost arena games for our-move eval drops using Pikafish.
- Built combined100 d5 failure analysis:
  `/home/laure/alphaxiang/v13_refutation_curriculum/analysis_022d_d5_combined100_depth7.json`
- Built and labeled combined100 repair slices:
  - `/home/laure/alphaxiang/v13_refutation_curriculum/combined100_opening_refutation_teacherq_d12`
  - `/home/laure/alphaxiang/v13_refutation_curriculum/combined100_verified_first_blunders_teacherq_d12`
  - `/home/laure/alphaxiang/v13_refutation_curriculum/combined100_post_blunder_after_value_policy_d10`
- Added pairwise teacher-Q loss to `xiangqi_train.py`:
  - `--teacher-q-pairwise-loss-weight`
  - `--teacher-q-pairwise-margin-logit`
  - `--teacher-q-pairwise-min-gap-cp`

## Arena Results

### 025b combined100 full mix

- Run:
  `/home/laure/alphaxiang/training_runs/run_025b_v13_combined100_refutation_fullmodel_anchor_from022d18000`
- d5 smoke:
  - step18200: `0W-4L-2D`, score `16.7%`
  - step18300: `1W-5L-0D`, score `16.7%`
- Verdict: reject. The broad opening/refutation mix hurts d5 relative to 022d.

### 025d verified blunders, no opening slice

- Run:
  `/home/laure/alphaxiang/training_runs/run_025d_v13_combined100_verified_noopening_rbuf256_anchor_from022d18000`
- Candidate:
  `/home/laure/alphaxiang/training_runs/run_025d_v13_combined100_verified_noopening_rbuf256_anchor_from022d18000/snapshots/latest_step18100.pt`
- d5 smoke:
  `/home/laure/alphaxiang/v13_refutation_curriculum/smoke_run025d_step18100_d5_sims8000_6g_seed202605721_cuda0/pika_d5/external_arena_20260508_042045.json`
  - `3W-3L-0D`, score `50.0%`
- d4 smoke:
  `/home/laure/alphaxiang/v13_refutation_curriculum/smoke_run025d_step18100_d4_sims8000_6g_seed202605722_cuda0/pika_d4/external_arena_20260508_045045.json`
  - `3W-2L-1D`, score `58.3%`
- Verdict: healthier than 025b, but does not beat 022d controls.

### 025f teacher-Q overfit diagnostic

- Run:
  `/home/laure/alphaxiang/training_runs/run_025f_v13_teacherq_overfit_diagnostic_noanchor_nohuman_from022d18000`
- Offline alignment improved strongly:
  - top1 agreement: `35.5% -> 43.9%`
  - top3 recall: `65.2% -> 72.3%`
  - median model top1 regret: `23cp -> 0cp`
- d5 smoke:
  `/home/laure/alphaxiang/v13_refutation_curriculum/smoke_run025f_step18100_d5_sims8000_6g_seed202605721_cuda0/pika_d5/external_arena_20260508_064907.json`
  - `0W-4L-2D`, score `16.7%`
- Verdict: confirms teacher-Q can move the model, but unconstrained repair destroys global strength.

### 025g pairwise teacher-Q with anchor

- Run:
  `/home/laure/alphaxiang/training_runs/run_025g_v13_pairwise_teacherq_verified_anchor_from022d18000`
- Best offline result at step18100:
  - top1 agreement: `34.2%`
  - top3 recall: `65.8%`
  - median regret: `17.5cp`
- d5 smoke:
  `/home/laure/alphaxiang/v13_refutation_curriculum/smoke_run025g_step18100_d5_sims8000_6g_seed202605721_cuda0/pika_d5/external_arena_20260508_074948.json`
  - `1W-3L-2D`, score `33.3%`
- Verdict: pairwise is safer than overfit CE, but still not a ship candidate.

### 025h stronger pairwise, no anchor top1 CE

- Run:
  `/home/laure/alphaxiang/training_runs/run_025h_v13_pairwise_stronger_no_top1_anchor_from022d18000`
- Best offline result at step18150:
  `/home/laure/alphaxiang/v13_refutation_curriculum/alignment_025h_step18150_combined100_verified_blunders.json`
  - top1 agreement: `36.8%`
  - top3 recall: `67.7%`
  - median regret: `15cp`
- Verdict: best offline pairwise result so far, but not promoted without arena proof.

## Interpretation

- The d5 failure-slice signal is real. A no-anchor overfit run can dramatically improve teacher-Q alignment.
- The same changes hurt actual arena strength, so the issue is not labeler wiring; it is preserving global policy while changing a few tactical decisions.
- Full teacher-Q CE is too style-forcing. Pairwise regret is directionally better, but current anchor design is still too blunt.

## Recommendation

- Freeze 022d as the practical v13 best checkpoint for now.
- Do not ship 025b/025d/025f/025g/025h.
- Keep the pairwise teacher-Q code; it is useful and compiles.
- Future research should use selective anchoring:
  - preserve anchor distribution on non-teacher-Q moves,
  - relax anchor only inside the labeled candidate set,
  - apply pairwise loss only when teacher gap is large,
  - require d4/d5 controls before any promotion.

