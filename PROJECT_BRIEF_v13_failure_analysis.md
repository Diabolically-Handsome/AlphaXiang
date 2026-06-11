# PROJECT BRIEF: v13 Failure Analysis

## Scope

Main checkpoint:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt`

Primary evidence:

- Pika d3 @8000, combined 50 games: `22W-10L-18D`, `62.0%`
- Pika d4 @6400, combined 50 games: `24W-14L-12D`, `60.0%`
- Pika d5 @8000, combined 50 games: `16W-20L-14D`, `46.0%`
- Fairy d3 @1600: `46W-2L-2D`, `94.0%`
- CNN-best @1600: `45W-3L-2D`, `92.0%`

Generated analysis artifacts:

- Depth-7 full loss scan: `/home/laure/alphaxiang/v13_failure_analysis/022d_pika_loss_eval_depth7.json`
- Depth-12 key blunder verification: `/home/laure/alphaxiang/v13_failure_analysis/022d_key_blunders_depth12.json`

## High-Level Finding

The failures do not look like random capacity-ceiling losses. They are clustered:

- Pika d4 losses are overwhelmingly black-side losses.
- Pika d5 losses split across red and black, but most belong to two repeatable opening families.
- Several d3/d4/d5 losses contain concrete tactical collapse points where Pikafish sees a defensive resource or mate threat that v13 misses.
- Many loss positions are already bad before the final visible blunder, so training on the whole loss trajectory is noisy. This explains why `run_023b` damaged strength.

## Loss Distribution

Pika losses analyzed:

- Total Pika losses: `44`
- Our-turn positions evaluated at Pika depth 7: `2298`
- Evaluation errors: `0`

By anchor:

| Anchor | Losses | Side Pattern |
|---|---:|---|
| Pika d3 | 10 | 5 red / 5 black |
| Pika d4 | 14 | 1 red / 13 black |
| Pika d5 | 20 | 9 red / 11 black |

Interpretation:

- d4 is the clearest black-side weakness.
- d5 is not just a color issue; it has both red tactical traps and black defensive structure problems.

## Opening Clusters

Top repeated Pika-loss opening families:

| Count | Opening Prefix |
|---:|---|
| 11 | `b2e2 b7e7 b0c2 b9c7 a0b0 h9g7 c3c4 g6g5` |
| 9 | `b2e2 b9c7 h2f2 h9g7 g3g4 i9i8 b0c2 c6c5` |
| 8 | `g3g4 b7e7 b0c2 c6c5 b2a2 b9c7 a0b0 h9g7` |
| 5 | `c3c4 b7e7 b0c2 b9c7 a0b0 a9b9 b2b6 h9g7` |

Interpretation:

- These are not one-off bad games.
- We should build an opening-conditioned failure suite from these prefixes and force both colors to play through them.

## Blunder Timing

Depth-7 scan classification:

| Class | Count | Meaning |
|---|---:|---|
| clear first drop | 14 | roughly playable before the move, then drops by >=250 cp |
| already bad before big drop | 23 | final collapse happens after the position is already strategically/tactically bad |
| no single big drop | 7 | slow squeeze or evaluation uncertainty |

Interpretation:

- Whole-loss-window training is too noisy.
- The next dataset should focus on first clear drops and pre-collapse windows, not full losing games.

## Key Verified Examples

Depth-12 checks:

| Anchor | Side | Game | Ply | Played | Pika Best | Before | After | Drop | Note |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| d5 | red | 0 | 38 | `f5g7` | `c2b0` | -397 | -829 | 432 | repeated red-side trap |
| d5 | black | 21 | 57 | `c7d5` | `c0c3` | 38 | -382 | 420 | playable position becomes bad |
| d5 | black | 25 | 81 | `e8d8` | `d3f4` | -296 | forced mate | huge | defensive resource missed |
| d3 | black | 35 | 51 | `e4e3` | `e4d4` | 44 | forced mate | huge | tactical danger blindness |
| d3 | red | 36 | 78 | `e1d0` | `e2c0` | 379 | forced mate | huge | winning/playable position collapses |

Some apparent depth-7 blunders weakened under depth-12:

- d5 black `g9g7`: depth 12 sees only a 102 cp drop, not a major blunder.
- d4 black `b9b6`: depth 12 sees only a 40 cp drop.
- d4 black `c9a7` / `e8f9`: medium drops, not decisive root causes.

Interpretation:

- We need depth-12/15 verification before adding a position to the high-weight training slice.
- The strongest actionable examples are the repeated d5 red trap, d5 black `c7d5/e8d8`, and d3 tactical collapses.

## Why run_023b Failed

`run_023b` used:

- d5 loss-slice ratio: `20%`
- teacher_q loss weight: `0.08`
- all d5 loss positions, including already-lost tails

Observed:

- step19000 d5 smoke: `2W-4L-4D`, `40.0%`
- step20000 partial: `0W-4L-0D` before stop

Root cause:

- Too much of the dataset was already-lost or late forced-mate material.
- The model was trained to fit losing-position noise instead of learning the earliest avoidable mistake.

## Recommended Next Data Plan

Build `v13_failure_v2` from verified high-value positions only:

1. First clear drops:
   - before score >= `-250 cp`
   - drop >= `250 cp`
   - include the position before the bad move and 1-2 earlier context plies

2. Repeated opening traps:
   - red d5 family around `f5g7`
   - black d5 family around `c7d5` / `e8d8`
   - black d4 family from the `g3g4 b7e7 ...` opening prefix

3. Tactical danger windows:
   - positions where depth-12 sees forced mate after the model move
   - include Pika best defensive move as a high-priority teacher_q candidate

Exclude:

- late forced-mate tails once the position is already below `-700 cp`
- repeated positions where depth-12 reduces the apparent blunder to <150 cp
- full loss trajectories

## Recommended Training Recipe

Keep `022d step18000` as main checkpoint and train only needle-like probes:

- new verified failure slice ratio: `1%-3%`
- teacher_q loss weight: `0.015-0.025`
- policy_oracle_alpha: `0.005-0.01`
- LR: `1e-6` to `2e-6`
- duration: `300-500` steps per probe
- keep v12.6 anchor distillation on

Promotion tests:

- Pika d3 @8000: must stay near `60%`
- Pika d4 @6400: must stay near `60%`
- Pika d5 @8000: must exceed `50%` in at least a 20-game smoke before larger panel

## Research Direction

This failure pattern supports the paper angle:

> Scaling the Transformer unlocks stronger global play, but tactical danger recognition and opening-conditioned defensive repair remain the next bottleneck.

The next useful innovation is not generic more data. It is a verified hard-position curriculum:

- opening-conditioned adversarial evaluation,
- first-blunder mining,
- depth-verified teacher_q,
- tiny-ratio curriculum to avoid overwriting the main policy.

