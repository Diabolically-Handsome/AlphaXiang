# v12.8H Full-Pika Refutation Teacher Results

Date: 2026-05-03

## Goal

Test whether the v12.8E global-strategy adapter failed at Pika d5 because the
tactical teacher was not strong enough.

## Code Changes

- `tools/action_value_labeler.py`
  - Added optional `--candidate-checkpoint`.
  - Added `--model-top-k`.
  - The labeler can now include the student's own top legal policy candidates
    in the candidate set before asking Pikafish to evaluate child positions.
- `tools/oversample_shard_run.py`
  - Creates an oversampled run directory by giving the same clean labeled shards
    multiple unique paths.
- Added v12.8H scripts:
  - `tools/_run_v128h_fullpika_refutation_data.sh`
  - `tools/_run_v128h_fullpika_refutation_train.sh`
  - `tools/_run_v128h_after_refutation_smokes.sh`

## Data Built

- Source arena:
  - `/home/laure/alphaxiang/v128_snapshot_smoke/global_strategy_anchor_step297000_quick/pika_d5/external_arena_20260503_071621.json`
- Student checkpoint:
  - `/home/laure/alphaxiang/training_runs/run_019f_v128_global_strategy_anchor/snapshots/latest_step297000.pt`
- Output:
  - `/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses`

Verified data:

- 18 lost d5 games used.
- 1159 positions extracted from our turns.
- Depth-16 oracle value: 1159/1159 labeled.
- Depth-10 MultiPV-8 oracle policy: 1159/1159 labeled.
- Hard mining marked 465/1159 positions.
- Depth-16 teacher-Q:
  - 465 hard rows labeled.
  - 4647 candidate child evals.
  - Candidate set included oracle top moves, played move, and student model top-6 legal moves.
- Hygiene audit:
  - dirty_shards = 0
  - oracle_illegal_entries = 0
  - teacher_q_illegal_entries = 0

## Training Arms

### H1: x16 Oversampled Full-Pika Refutation

- Data:
  - `/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses_x16`
  - 48 shard paths, 18544 samples.
- Run:
  - `/home/laure/alphaxiang/training_runs/run_019i_v128h_fullpika_refutation_x16`
- Params:
  - resume v12.8E step297000
  - train global adapter only
  - teacher_q_loss_weight = 0.30
  - policy_oracle_alpha = 0.15
  - anchor_policy_kl = 0.07
  - max_steps = 298000

Smoke results:

- step297500:
  - d3 5-6-1 = 45.8%
  - d4 stopped after d3 failed
- step298000:
  - d3 3-7-2 = 33.3%
  - d4 stopped after d3 failed

Conclusion: too strong; destabilized the public anchors.

### H2: x4 Guarded Full-Pika Refutation

- Data:
  - `/home/laure/alphaxiang/v128h_fullpika_refutation_data/v128e_d5_losses_x4`
  - 12 shard paths, 4636 samples.
- Run:
  - `/home/laure/alphaxiang/training_runs/run_019j_v128h_fullpika_refutation_x4_guarded`
- Params:
  - resume v12.8E step297000
  - train global adapter only
  - teacher_q_loss_weight = 0.12
  - policy_oracle_alpha = 0.05
  - anchor_policy_kl = 0.12
  - max_steps = 297500

Smoke results:

- d3 7-1-4 = 75.0%
- d4 4-8-0 = 33.3%
- Gate stopped before d5.

Conclusion: strong d3 spike, but d4 collapse. This is shallow exploitation, not a ship candidate.

## Interpretation

- The "stronger teacher" idea is valid, but the naive direct finetune is not.
- Full-Pika refutation has enough signal to move behavior quickly.
- However, when trained only on d5 losses, the adapter shifts style instead of
  learning a stable cross-depth tactical prior.
- This is the same failure family as memory/q-weight: one anchor improves while
  another anchor drops.

## Recommendation

- Do not ship v12.8H H1/H2.
- Keep the full-Pika refutation dataset; it is clean and valuable.
- Next attempt should not simply increase teacher strength or token count.
- If continuing v12:
  - add a stratified sampler or explicit balanced replay so every batch has:
    - d3/d4 preservation positions,
    - d5 refutation positions,
    - normal stage2 distribution;
  - or use full-Pika teacher-Q only as a low-frequency regularizer with an
    explicit d3/d4 preservation loss.
- If moving to v13:
  - carry global strategic tokens forward as a built-in architecture feature,
  - use the full-Pika refutation data as a diagnostic/auxiliary curriculum,
  - do not use it as the dominant training distribution.
