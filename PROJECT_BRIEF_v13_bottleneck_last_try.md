# V13 Bottleneck And Last-Try Repair Brief

Date: 2026-05-09

## One-Line Conclusion

V13 is not weak in ordinary positions. The best verified checkpoint still appears to be:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`

The final repair attempt did not beat it. The current bottleneck is most likely rule-state/data cleanliness plus high-pressure tactical refutation and endgame conversion, not simple model capacity or more generic steps.

## Verified Strength After Long-Check Rule Fix

The long-check/repetition adjudication bug was real and has been fixed in `xqcpp_ext_hist8_115.cpp`, but it was not the sole reason V13 was strong at lower anchors.

Using V13 022d step18000, sims=8000, scalar value:

- Pika d3, 20 games: `15W-2L-3D`, score `82.5%`
  - JSON: `/home/laure/alphaxiang/v13_repetition_rule_fix_smoke/step18000_sims8000_20g/pika_d3/external_arena_20260509_014708.json`
- Pika d4, 6 games: `3W-2L-1D`, score `58.3%`
  - JSON: `/home/laure/alphaxiang/v13_repetition_rule_fix_smoke/step18000_sims8000_quick_d4d5_6g/pika_d4/external_arena_20260509_021718.json`
- Pika d5, 6 games: `1W-5L-0D`, score `16.7%`
  - JSON: `/home/laure/alphaxiang/v13_repetition_rule_fix_smoke/step18000_sims8000_quick_d4d5_6g/pika_d5/external_arena_20260509_024221.json`

Interpretation: d3/d4 remain healthy after the rule fix; d5 still catches tactical/mate failures.

## Final Repair Attempt

I added a new training option in `xiangqi_train.py`:

- `--bad-move-suppression-loss-weight`
- `--bad-move-suppression-margin-logit`
- `--bad-move-suppression-min-gap-cp`
- `--bad-move-suppression-beta`

Purpose: suppress known `bad_move` actions from the d5 failure shard relative to frozen V13, without full teacher CE imitation.

Repair data:

`/home/laure/alphaxiang/v13_refutation_curriculum/combined100_verified_first_blunders_teacherq_d12/train/shard_000000.pt`

Key data facts:

- 155 samples
- all 155 have `bad_move`
- 40 samples have teacher-Q gap >= 80cp for the known bad move

## Last-Try Results

### 027a: Full-model log-prob suppression

Output:

`/home/laure/alphaxiang/training_runs/run_027a_v13_badmove_suppression_anchor_from022d18000`

Offline diagnostic:

`/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_027a_vs_022d_combined100_verified.json`

Result:

- policy KL mean `0.349`
- top1 change `15.5%`
- known bad move rank stayed `1.0 -> 1.0`
- known bad log-prob slightly increased

Conclusion: too weak to actually demote the known bad moves.

### 027b: Policy-head-only log-prob suppression

Output:

`/home/laure/alphaxiang/training_runs/run_027b_v13_policyhead_badmove_suppression_from022d18000`

Offline diagnostics:

- `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_027b_step18050_vs_022d_combined100_verified.json`
- `/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_027b_step18100_vs_022d_combined100_verified.json`

Result:

- median regret improved by `-18.5cp` at step18050 and `-10.0cp` at step18100
- known bad move rank stayed `1.0 -> 1.0`
- known bad log-prob did not reliably decrease

Conclusion: improves a soft metric, but does not solve the concrete bad top1 problem.

### 027c: Policy-head-only raw bad-logit suppression

Output:

`/home/laure/alphaxiang/training_runs/run_027c_v13_policyhead_badlogit_suppression_from022d18000`

Offline diagnostic:

`/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_027c_step18050_vs_022d_combined100_verified.json`

Result:

- known bad probability decreased
- known bad log-prob decreased
- known bad rank still stayed `1.0 -> 1.0`
- policy KL mean rose to `1.214`
- top1 change rose to `47.7%`

Arena smoke against Pika d5:

- `0W-5L-1D`, score `8.3%`
- JSON: `/home/laure/alphaxiang/v13_last_try_027c_smoke/d5_sims8000_6g/external_arena_20260509_061311.json`

Conclusion: this did suppress bad moves somewhat, but it damaged the policy distribution and performed worse than 022d.

### 027d: Balanced bad-logit suppression

Output:

`/home/laure/alphaxiang/training_runs/run_027d_v13_policyhead_badlogit_balanced_from022d18000`

Offline diagnostic:

`/home/laure/alphaxiang/v13_refutation_curriculum/diagnostic_027d_step18050_vs_022d_combined100_verified.json`

Result:

- median regret improved by `-22cp`
- policy KL mean still high at `0.906`
- top1 change still high at `44.5%`
- known bad rank stayed `1.0 -> 1.0`

Conclusion: still too much policy drift for too little concrete bad-rank improvement.

## Current Bottlenecks

### 1. Rule-State And Toxic Historical Labels

The engine now detects many old repetition draws as long-check losses. This means some old value targets were objectively polluted.

Important nuance: the neural input has 8-frame history plus repetition/no-capture planes, so it is not completely history-blind. However, it still does not explicitly encode responsibility for repetition, continuous checking side, or long-check legality state. Those rule states matter in Xiangqi.

### 2. Pika d5 Tactical Refutation

d3/d4 are healthy, while d5 still wins by mate in most losses. That points to local forcing lines:

- king safety
- consecutive checks
- mate threats
- defended tactical sacrifices
- moves that look globally good but are tactically refuted

### 3. Endgame Conversion And No-Capture Pressure

V13 can reach good positions but sometimes does not convert efficiently. no-capture draws still appear in d3/d4/d5 testing. This suggests an endgame conversion suite is needed.

### 4. Local Repair Causes Global Drift

DPO-style and bad-move-only repair can improve offline regret, but the bad move often remains top1. When the repair is made strong enough to affect bad moves, policy KL/top1 drift becomes large and arena gets worse.

This is the main lesson of the final attempt: checkpoint finetuning is not precise enough for these V13 tactical wounds.

## Recommendation

Do not ship 027a/027b/027c/027d.

Keep 022d step18000 as the best verified V13 checkpoint for now.

The next serious attempt should not be another small failure-slice finetune. I recommend:

1. Detox old data affected by repetition/long-check/no-capture adjudication.
2. Make MCTS state and any transposition key explicitly rule-state-aware.
3. Add a search-side tactical safety layer at root or high-risk nodes, especially for mate/check/capture forcing refutations.
4. Build d5 loss clustering plus an endgame conversion suite before training again.
5. If training resumes, use localized pairwise repair only after the above rule/search fixes, with strict anchor drift gates.

Postmortem sentence:

V13 learned a strong global policy, but the remaining ceiling is controlled by rule-state correctness, high-depth tactical safety, and endgame conversion rather than ordinary loss minimization.
