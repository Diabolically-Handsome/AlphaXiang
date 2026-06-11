# AlphaXiang V14 Failure Report

Date: 2026-05-14

## Executive Summary

V14 did not fail because the CNN+Transformer idea is obviously wrong. It failed because every attempted integration made the CNN answer the wrong question.

The successful V13 behavior comes from a strong global Transformer policy plus high-sim MCTS. V13's remaining weakness is narrow: high-pressure forcing tactics, black-side king safety, and d5/d6 conversion under stronger Pikafish. V14 tried to inject local CNN bias into either board tokens, policy logits, or root move selection. None of those routes reliably produced executable tactical corrections.

The repeated failure pattern:

1. CNN adapters can move the policy, but they do not know which tactical changes are safe.
2. Pairwise repair can suppress known bad moves, but the effect does not generalize enough to d5/d6.
3. The action-danger head learns an offline risk signal, but online it mostly fires alarms without finding legal, provably better replacement moves.
4. Full hybrid retraining from scratch looks healthy by validation loss, but remains a novice in arena after thousands of steps.
5. Gumbel-style root randomization increases variance and hurts defensive stability when used directly.

The safest final conclusion: keep V13/run031a as the release model. Do not ship any V14 checkpoint.

## Baseline Context

Current release candidate:

```text
/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt
```

Public reference note:

```text
PROJECT_BRIEF_public_elo_reference.md
```

That note estimates V13.3 around 2350-2450 public-engine-anchor Elo, with 2400 as the clean public number. V13/run031a improved the known black-side d5 behavior relative to earlier V13 checkpoints, while Pika d6 still exposed high-pressure tactical and conversion weaknesses.

## V14A: Local Tactical CNN Adapter

### Hypothesis

Add a small CNN adapter to the existing V13 board-token stream. The CNN acts as a local tactical eye while the Transformer keeps its global policy.

### Implementation

Training run:

```text
/home/laure/alphaxiang/training_runs/run_040a_v14a_cnn_local_adapter_from031a19000
```

Script:

```text
tools/_run_v14a_cnn_local_adapter.sh
```

Key settings:

```text
base = run031a step19000
model_preset = v14a_200m_cnn_adapter
train_only_cnn_local_adapter = true
learning_rate = 3e-6
anchor_policy_kl_weight = 2.0
anchor_value_mse_weight = 0.5
normal / preservation / refutation data = 0.70 / 0.10 / 0.20
```

### Arena Evidence

All tests used 8000 sims.

| Test | W-L-D | Score | Evidence |
|---|---:|---:|---|
| Pika d3 model-only | 4-1-1 / 6 | 75.0% | `/home/laure/alphaxiang/v14a_snapshot_smoke/step20000/pika_d3_g6_model_only/external_arena_20260511_061904.json` |
| Pika d4 model-only | 1-3-2 / 6 | 33.3% | `/home/laure/alphaxiang/v14a_snapshot_smoke/step20000/pika_d4_g6_model_only/external_arena_20260511_064123.json` |
| Pika d5 model-only | 0-1-5 / 6 | 41.7% | `/home/laure/alphaxiang/v14a_snapshot_smoke/step20000/pika_d5_g6_model_only/external_arena_20260511_070511.json` |
| Pika d5 ship config | 2-3-1 / 6 | 41.7% | `/home/laure/alphaxiang/v14a_snapshot_smoke/step20000/pika_d5_g6_ship/external_arena_20260511_074055.json` |

### Failure Diagnosis

V14A did not collapse at d3, but it damaged d4 and did not improve d5. The adapter was too late and too weak: it was attached after V13 had already formed a stable global representation, then trained in isolation under strong anchor KL. That setup allowed only tiny local perturbations, not a new tactical reasoning mechanism.

In plain terms: the CNN could decorate V13's board tokens, but it could not change the search-critical mistakes in a targeted way.

## V14B: CNN Policy Residual Adapter

### Hypothesis

If adding CNN context into board tokens is too indirect, let CNN write directly into policy logits as a residual correction.

### Implementation

Training run:

```text
/home/laure/alphaxiang/training_runs/run_041a_v14b_cnn_policy_residual_from031a19000
```

Script:

```text
tools/_run_v14b_cnn_policy_residual.sh
```

Key settings:

```text
base = run031a step19000
model_preset = v14b_200m_cnn_policy_residual
train_only_cnn_policy_residual_adapter = true
learning_rate = 1e-6
anchor_policy_kl_weight = 3.0
anchor_value_mse_weight = 0.25
```

### Arena Evidence

| Test | W-L-D | Score | Evidence |
|---|---:|---:|---|
| Pika d3 model-only | 4-1-1 / 6 | 75.0% | `/home/laure/alphaxiang/v14b_snapshot_smoke/step19250/pika_d3_g6_model_only/external_arena_20260511_082758.json` |
| Pika d4 model-only | 3-1-2 / 6 | 66.7% | `/home/laure/alphaxiang/v14b_snapshot_smoke/step19250/pika_d4_g6_model_only/external_arena_20260511_085026.json` |
| Pika d5 model-only | 1-3-2 / 6 | 33.3% | `/home/laure/alphaxiang/v14b_snapshot_smoke/step19250/pika_d5_g6_model_only/external_arena_20260511_091342.json` |
| Pika d5 ship config | 1-2-3 / 6 | 41.7% | `/home/laure/alphaxiang/v14b_snapshot_smoke/step19250/pika_d5_g6_ship/external_arena_20260511_094742.json` |

### Failure Diagnosis

V14B showed that direct policy residuals can move the model, but not in the right direction. d4 looked promising in a tiny sample, but d5 became worse than the target problem. This is consistent with policy interference: the CNN residual is powerful enough to perturb V13's move ordering, but it lacks reliable evidence about which tactical alternative survives search.

The problem was not "CNN has no signal"; the problem was "CNN signal is not grounded enough to edit final policy logits."

## V14C: CNN Policy Residual + Pairwise Bad-Move Repair

### Hypothesis

Avoid broad policy imitation. Train the CNN residual surgically: suppress known bad moves and raise teacher-Q-preferred alternatives while anchoring to V13.

### Implementation

Training run:

```text
/home/laure/alphaxiang/training_runs/run_041b_v14c_cnn_policy_pairwise_from031a19000
```

Script:

```text
tools/_run_v14c_cnn_policy_pairwise.sh
```

Key settings:

```text
base = run031a step19000
model_preset = v14b_200m_cnn_policy_residual
train_only_cnn_policy_residual_adapter = true
learning_rate = 3e-7
policy_loss_weight = 0.15
teacher_q_pairwise_loss_weight = 0.40
teacher_q_pairwise_bad_move_only = true
bad_move_suppression_loss_weight = 0.15
anchor_policy_kl_weight = 4.0
anchor_value_mse_weight = 0.25
```

### Arena Evidence

| Test | W-L-D | Score | Evidence |
|---|---:|---:|---|
| Pika d4 model-only | 1-1-4 / 6 | 50.0% | `/home/laure/alphaxiang/v14c_snapshot_smoke/step19250/pika_d4_g6_model_only/external_arena_20260511_101950.json` |
| Pika d5 model-only | 2-3-1 / 6 | 41.7% | `/home/laure/alphaxiang/v14c_snapshot_smoke/step19250/pika_d5_g6_model_only/external_arena_20260511_103413.json` |
| Pika d5 ship config | 1-2-3 / 6 | 41.7% | `/home/laure/alphaxiang/v14c_snapshot_smoke/step19250/pika_d5_g6_ship/external_arena_20260511_110936.json` |

### Failure Diagnosis

V14C preserved the model better than V14B, but still did not create a breakthrough. Pairwise suppression can fix the exact known bad move family, but d5/d6 failures are not a single reusable bad-move pattern. They are context-dependent forcing and conversion failures.

Known bad moves are labels for yesterday's mistakes. They are not a general tactical oracle.

## V14D: Action-Conditioned CNN Danger Head

### Hypothesis

Do not let CNN play chess. Let CNN evaluate candidate moves after they are applied:

```text
state s + candidate move m -> resulting state s' -> CNN danger score
```

Then use danger to rerank root candidates.

### Implementation

Dataset:

```text
/home/laure/alphaxiang/v14d_danger_data/danger_dataset.pt
```

Dataset report:

```text
/home/laure/alphaxiang/v14d_danger_data/danger_dataset.report.json
```

Training run:

```text
/home/laure/alphaxiang/training_runs/run_042a_v14d_action_danger_head
```

Summary:

```text
/home/laure/alphaxiang/training_runs/run_042a_v14d_action_danger_head/summary.json
```

Dataset size:

```text
127 candidate samples
26 groups
82 tactical_refuted positives
81 value_collapse positives
```

Best offline validation:

```text
val AUC = 0.732
val recall@0.5 = 0.789
val precision@0.5 = 0.789
val false_positive_rate@0.5 = 0.400
val pair_accuracy = 0.742
```

### Arena Evidence

| Test | W-L-D | Score | Events | Evidence |
|---|---:|---:|---:|---|
| V13 + danger rerank, d5 black-only | 1-3-2 / 6 | 33.3% | 34 | `/home/laure/alphaxiang/v14d_snapshot_smoke/run031a_step19000_danger_d5_black_only/external_arena_20260511_124317.json` |

### Failure Diagnosis

The offline danger head looked real, but online rerank was unsafe. It fired 33 CNN rerank events and still lost 3 of 6 games, with 4 mate terminations.

The key failure: a risk score is not a replacement move. Lowering or reranking root candidates based on a learned danger scalar can remove the original move without proving the replacement is safe.

V14D confirmed the right conceptual direction, but the wrong execution method.

## V14E: CNN Danger Triage + Exact Tactical Guard

### Hypothesis

Use CNN only as a fast alarm. If CNN says the selected move is dangerous, run exact mate/forcing-check guard. Do not let CNN directly rerank.

### Arena Evidence

| Test | W-L-D | Score | Events | Evidence |
|---|---:|---:|---:|---|
| threshold 0.90, d5 black-only | 1-2-3 / 6 | 41.7% | 5 | `/home/laure/alphaxiang/v14e_snapshot_smoke/run031a_step19000_triage_d5_black_only/external_arena_20260511_133338.json` |
| threshold 0.70, d5 black-only | 1-2-3 / 6 | 41.7% | 283 | `/home/laure/alphaxiang/v14e_snapshot_smoke/run031a_step19000_triage070_d5_black_only/external_arena_20260511_141207.json` |

Additional V13 8000-sim probe:

| Test | W-L-D | Score | Notes |
|---|---:|---:|---|
| V13 baseline, d5 black-only, 8000 sims | 0-1-3 / 4 | 37.5% | `/home/laure/alphaxiang/v13_search_hybrid_smoke/baseline_d5_black_8000/external_arena_20260512_035135.json` |
| V13 + danger triage, d5 black-only, 8000 sims | 0-1-1 / 2 | 25.0% | 51 triage events, 0 actual replacements; `/home/laure/alphaxiang/v13_search_hybrid_smoke/triage_oracle_d5_black_8000/external_arena_20260512_043915.json` |

### Failure Diagnosis

This was the most diagnostic failure.

At threshold 0.70, CNN danger triggered constantly, but the exact guard almost never found a concrete refutation and did not replace moves. In the later V13 probe, it fired 51 times and produced 0 changes.

That means the danger head learned "this smells risky," but not "this exact move loses by this exact forcing line." The exact oracle was also too narrow: mate1/mate2/check-only forcing lines do not cover all d5/d6 tactical losses.

V14E failed because alarm without executable proof is not enough.

## V14R: Restarted Trunk-Native Hybrid From Scratch

### Hypothesis

Maybe the problem is late CNN insertion. Train a new hybrid from scratch with a non-zero CNN local tactical stem, while using V13 only as a short teacher/anchor to cross the novice phase.

### Implementation

Training runs:

```text
/home/laure/alphaxiang/training_runs/run_050a_v14r_200m_hybrid_bootcamp
/home/laure/alphaxiang/training_runs/run_050b_v14r_200m_hybrid_strong_bootcamp
```

Script:

```text
tools/_run_v14r_hybrid_bootcamp.sh
```

V14R architecture:

```text
model_preset = v14r_200m_hybrid
use_cnn_local_tactical_stem = true
trained from scratch
V13 checkpoint used as teacher/anchor, not as initial weights
```

### Training Status

Run 050a:

```text
global_step = 20000
last_human_val_total_loss = 3.0254
```

Run 050b:

```text
global_step = 5363
interrupted = true, received SIGTERM
last_human_val_total_loss = 3.8599
```

### Arena Evidence

| Test | W-L-D | Score | Evidence |
|---|---:|---:|---|
| 050b step1000 vs Pika d1 | 0-4-0 / 4 | 0.0% | `/home/laure/alphaxiang/v14r_snapshot_smoke/050b_step1000_d1_quick/external_arena_20260512_014950.json` |
| 050a step12000 vs Pika d1 | 0-5-1 / 6 | 8.3% | `/home/laure/alphaxiang/v14r_snapshot_smoke/step12000_d1d2_quick/pika_d1/external_arena_20260512_010250.json` |
| 050a step12000 vs Pika d2 | 0-6-0 / 6 | 0.0% | `/home/laure/alphaxiang/v14r_snapshot_smoke/step12000_d1d2_quick/pika_d2/external_arena_20260512_010714.json` |

### Failure Diagnosis

V14R is the strongest evidence that lower validation loss is not enough. The model was learning the supervised distribution, but arena strength stayed near novice against Pika d1/d2.

Likely causes:

1. Scratch hybrid needs a much longer AlphaZero-style bootstrapping schedule than the weekend budget allowed.
2. V13 distillation teaches surface policy/value behavior, but not necessarily search-compatible internal calibration.
3. The CNN stem changes the early representation enough that MCTS value/policy calibration is poor even while supervised losses improve.
4. The training distribution is still dominated by teacher imitation and old selfplay, not a live curriculum where the new hybrid learns from its own MCTS failures.

V14R did not prove the hybrid architecture is impossible. It proved that a short scratch bootcamp cannot replace the months of curriculum that created V13.

## Gumbel Root Probe

### Hypothesis

Use Gumbel-style root selection to improve exploration among high-visit candidates, then combine it with CNN danger and exact tactical guards.

### Implementation

Added a lightweight root-level probe:

```text
--our-root-selection-mode gumbel_visit
--our-root-gumbel-top-k
--our-root-gumbel-scale
```

Code:

```text
tools/external_arena.py
```

This is not full Gumbel MuZero. It is only root visit Gumbelization after normal MCTS.

### Evidence

The direct Gumbel hybrid was stopped early:

```text
/home/laure/alphaxiang/v13_search_hybrid_smoke/hybrid_d5_black_8000/INTERRUPTED_NOTES.md
```

Observed before stop:

```text
game 1: loss, 137 plies, mate
game 2: loss, 131 plies, mate
```

### Failure Diagnosis

Direct Gumbel final-move replacement increased variance and hurt black-side defense. In Xiangqi tactical positions, the highest-visit root move is often chosen because it avoids hidden danger. Randomized replacement among top candidates can step into exactly the tactical refutations V13 is trying to avoid.

Gumbel should not be used as final root move selection in this system. If revisited, it should only be used during search expansion or as a replacement chooser after an exact safety veto, not as a direct policy override.

## Cross-Experiment Root Cause

### 1. The CNN Was Asked To Play, Not To Prove Safety

V14A/B/C all asked the CNN to influence policy. That is too hard. The CNN does not merely need to know "this local shape looks tactical." It must know:

```text
If we play move m, can the opponent force a win or severe collapse?
If yes, which line proves it?
Which alternative avoids that line without destroying the global plan?
```

The attempted CNN modules did not have enough supervision to answer that.

### 2. Offline Tactical Labels Were Too Small And Too Narrow

V14D's danger dataset had only:

```text
127 candidate samples
26 position groups
```

That is enough to get a plausible validation AUC, but not enough to generalize across Pika d5/d6 black-side defensive failures.

The data also mixed value collapse, known bad moves, mate labels, and tactical refutation labels. Those are related, but not identical. A move can look risky without allowing a short exact mate. That is exactly what happened online.

### 3. Exact Tactical Oracle Was Too Narrow

The current exact layer mainly catches:

```text
mate1
check-forced mate2
bounded check-only forcing lines
```

But V13's d5/d6 losses include more than check-only mates:

- defensive overloads,
- material collapse after a quiet threat,
- king-safety weakening that becomes fatal several plies later,
- endgame conversion failures,
- no-capture / repetition pressure,
- opening-family traps.

So CNN triage often fired, but exact guard found no replacement.

### 4. Late CNN Insertion Conflicted With A Mature Transformer

V13 already has a coherent policy distribution. Small CNN adapters trained after the fact can only perturb it. If the perturbation is weak, nothing changes. If it is strong, it damages global move ordering.

This explains the repeated pattern:

```text
small adapter -> no breakthrough
direct residual -> d5/d4 Pareto tradeoff
pairwise/suppression -> safer but still no generalization
```

### 5. High-Sim MCTS Remains Essential

The final V13 search probe showed that sims matter strongly:

```text
V13 d5 black-only, 3200 sims: looked much worse
V13 d5 black-only, 8000 sims: 0W-1L-3D / 4 = 37.5%
```

That means part of the "CNN not helping" result is also a search-calibration issue. V13 needs enough visits to let its global policy/value work. A CNN shortcut cannot cheaply replace those visits unless it is a much stronger tactical oracle.

## Final Verdict

V14 did not produce a ship candidate.

Best release model remains:

```text
run031a step19000
```

V14's negative result is still scientifically useful:

> Late-fused CNN local adapters do not reliably improve a strong Xiangqi Transformer under MCTS. Tactical weaknesses are better treated as search-state, tactical-oracle, and data-labeling problems than as generic local-convolution policy corrections.

## What To Write In The Paper

Positive framing:

```text
We explored several CNN-Transformer hybridization strategies after identifying high-pressure tactical failures in V13. Late local CNN adapters, policy residual adapters, pairwise bad-move suppression, and action-conditioned danger heads did not improve the release model. These negative results suggest that local convolutional inductive bias alone is insufficient once a Transformer policy has already learned strong global board representations. In Xiangqi, the remaining failures require executable tactical verification rather than generic local pattern recognition.
```

Short lesson:

```text
Transformer gives global policy strength; CNN does not automatically supply tactical correctness. For Xiangqi, local tactical repair must be tied to exact or search-verifiable refutations.
```

## Recommended Next Step

Do not continue V14 immediately.

Release V13.3, then write the paper around:

1. Transformer scaling and global strategy in Xiangqi.
2. Search-budget dependence of larger Transformer policies.
3. Tactical failure analysis at Pika d5/d6.
4. Negative hybrid results as future work.

If CNN+Transformer is revisited later, it should be a new project:

```text
V15 candidate:
Train a trunk-native hybrid from scratch with live selfplay, exact tactical labels, and a much larger action-conditioned refutation dataset.
```

Not a weekend patch on top of V13.
