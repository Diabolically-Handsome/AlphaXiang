# V14S Phase 1 Search Tuning

Date: 2026-05-14

## Goal

Do not change model weights.  Tune the search behavior of the fixed V13.3 release
candidate:

```text
/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt
```

The first target is Pika d5 black-side defense, because previous V13/V14 evidence
showed that this cell exposes the most repeatable tactical pressure while still
being close enough for search improvements to matter.

## Added Tools

```text
tools/v14s_search_tuning_grid.py
tools/_run_v14s_phase1_coarse_grid.sh
```

The runner wraps `tools/external_arena.py` and keeps the current V13.3 inference
safety stack:

```text
--our-root-mate1-blunder-guard
--our-tactical-mate1-extension
--our-tactical-mate2-extension
```

It writes:

```text
v14s_search_grid_summary.json
v14s_search_grid_summary.md
```

## First Coarse Grid

Started:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_coarse_d5_black_20260514_005915
PID: 198711
```

Configuration:

```text
opponent = Pikafish depth 5
AlphaXiang side = black
sims = 8000
games per combo = 2
temperature_move = 0.02
q_weight = 1.0
q_clip = 1.0
c_puct = 1.0, 1.25, 1.5
parallel_games = 1
```

This is a direction-finding smoke only.  Any apparent winner needs expansion to
at least 20 games, then red/black split and d6 checks.

## First Results

Completed:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_coarse_d5_black_20260514_010155_setsid
```

Pika d5, AlphaXiang black, 8000 sims, 2 games per config:

| c_puct | W-L-D | score | terminations |
|---:|---:|---:|---|
| 1.00 | 0-2-0 | 0.0% | 2 mate losses |
| 1.25 | 0-1-1 | 25.0% | 1 mate, 1 nocap |
| 1.50 | 0-0-2 | 50.0% | 2 nocap |

Directional interpretation: lower `c_puct=1.0` looks too policy/value-greedy and
allows mate losses.  Around `1.4-1.5` looks more defensive.

Expanded directional check:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_expand_cpuct_d5_black_20260514_032531_setsid
```

Pika d5, AlphaXiang black, 8000 sims, 4 games per config:

| c_puct | W-L-D | score | terminations |
|---:|---:|---:|---|
| 1.40 | 0-0-4 | 50.0% | 4 nocap |
| 1.50 | 0-0-4 | 50.0% | 4 nocap |
| 1.60 | 0-3-1 | 12.5% | 3 mate, 1 nocap |

Important caveat: with standard start position, fixed side, no root noise, and no
opening suite, repeated games for the same config can become identical or nearly
identical.  These cells are therefore not Elo evidence.  They are directional
debug evidence that `c_puct` has a real effect and that the useful range is likely
below the failure point around `1.6`.

Next required step: add or use an opening-suite path for `external_arena.py`, then
retest `c_puct=1.35/1.40/1.45/1.50` across diverse fixed openings.

## Opening Suite Support

Implemented in:

```text
tools/external_arena.py
tools/v14s_search_tuning_grid.py
tools/_run_v14s_phase1_coarse_grid.sh
```

New external arena flags:

```text
--opening-suite-path
--games-per-opening
--max-openings
```

Smoke passed:

```text
/home/laure/alphaxiang/v14s_search_tuning/opening_smoke_one/external_arena_20260514_064307.json
```

Started diversified opening-suite grid:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_opening_suite_d5_black_20260514_064406_setsid
PID: 203044
```

Configuration:

```text
opening_suite = arena_openings/human_val_opening_suite_v1.json
max_openings = 4
games_per_opening = 1
opponent = Pikafish d5
AlphaXiang side = black
sims = 8000
c_puct = 1.35, 1.40, 1.45, 1.50
q_weight = 1.0
temperature_move = 0.02
```

Result:

```text
c_puct 1.35: 0-3-1 / 4 = 12.5%
c_puct 1.40: 1-2-1 / 4 = 37.5%
c_puct 1.45: 3-1-0 / 4 = 75.0%
c_puct 1.50: 1-2-1 / 4 = 37.5%
```

Evidence:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_opening_suite_d5_black_20260514_064406_setsid/v14s_search_grid_summary.md
```

Interpretation: `c_puct=1.45` is the first real V14S candidate.  The sample is
still tiny, but the result survived diversified openings and is much cleaner
than the fixed-start smoke.

Started 12-opening validation against the current baseline:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_validate_cpuct145_d5_black_20260514_085318_setsid
PID: 204587
```

Configuration:

```text
max_openings = 12
games_per_opening = 1
c_puct = 1.25, 1.45
opponent = Pikafish d5
AlphaXiang side = black
sims = 8000
```

Status update: this comparison was stopped during the first `c_puct=1.25` cell
before any JSON result was written.  Reason: existing evidence already showed
`1.25` as a poor baseline and the run was consuming time without producing a
decision-relevant result.  The directory is preserved as an aborted run, not as a
completed comparison.

Promoted provisional candidate for direct validation:

```text
c_puct = 1.45
q_weight = 1.0
q_clip = 1.0
temperature_move = 0.02
sims = 8000
ship safety = root mate1 guard + mate1/mate2 extension
```

Started 1.45-only 12-opening validation:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_validate_cpuct145_only_d5_black_20260514_090056_setsid
PID: 205025
```

Configuration:

```text
max_openings = 12
games_per_opening = 1
c_puct = 1.45
opponent = Pikafish d5
AlphaXiang side = black
sims = 8000
```

Result:

```text
5W-5L-2D / 12 = 50.0%
avg plies = 99.8
terminations = mate 8, nocap 2, longcheck 2
json = /home/laure/alphaxiang/v14s_search_tuning/phase1_validate_cpuct145_only_d5_black_20260514_090056_setsid/sims8000_cpuct1.45_q1_clip1_temp0.02_black/external_arena_20260514_101658.json
summary = /home/laure/alphaxiang/v14s_search_tuning/phase1_validate_cpuct145_only_d5_black_20260514_090056_setsid/v14s_search_grid_summary.md
```

Per opening:

| opening | id | result | plies | termination |
|---:|---|---|---:|---|
| 0 | `val_shard_00070_1342` | our_win | 76 | mate |
| 1 | `val_shard_00070_2381` | opp_win | 71 | mate |
| 2 | `val_shard_00043_2220` | our_win | 80 | mate |
| 3 | `val_shard_00043_0124` | our_win | 96 | mate |
| 4 | `val_shard_00028_3427` | our_win | 34 | mate |
| 5 | `val_shard_00028_0000` | opp_win | 159 | mate |
| 6 | `val_shard_00012_2126` | draw | 114 | nocap |
| 7 | `val_shard_00012_2844` | draw | 141 | nocap |
| 8 | `val_shard_00023_3664` | our_win | 111 | longcheck |
| 9 | `val_shard_00023_3515` | opp_win | 147 | mate |
| 10 | `val_shard_00057_2806` | opp_win | 103 | mate |
| 11 | `val_shard_00057_0661` | opp_win | 66 | longcheck |

Interpretation: `c_puct=1.45` did not reproduce the 75% score from the 4-opening
smoke, but it did convert the previously weak black-side Pika d5 cell into a
roughly even 12-opening result.  This is still not a release-level conclusion:
standard error is about 14pp, and the loss pattern still contains mate/longcheck
failures.  However, it is the best search-only candidate so far and should be
kept as the provisional V14S setting unless a nearby local grid beats it.

## Log-Scaled cPUCT / FPU Implementation

Implemented new search-only knobs:

```text
xqcpp_ext_hist8_115.cpp
xiangqi_mcts_ext.py
tools/external_arena.py
tools/v14s_search_tuning_grid.py
tools/_run_v14s_phase1_coarse_grid.sh
```

New MCTS parameters:

```text
--our-c-puct-base
--our-c-puct-factor
--our-fpu-reduction-root
--our-fpu-reduction-tree
```

Default behavior is legacy-compatible:

```text
c_puct_factor = 0.0      -> fixed c_puct exactly as before
fpu_reduction_* = -1.0   -> unvisited child Q=0 exactly as before
```

When enabled:

```text
c_eff = c_puct + c_puct_factor * log((parent_visits + c_puct_base + 1) / c_puct_base)
```

FPU uses `parent_value - reduction` for unvisited children only when the
corresponding root/tree reduction is non-negative.

Compatibility checks passed:

```text
python -m py_compile xiangqi_mcts_ext.py tools/external_arena.py tools/v14s_search_tuning_grid.py
minimal C++ mcts_search legacy path
minimal C++ mcts_search logcpuct/FPU path
external_arena smoke with logcpuct/FPU flags
```

Started first log-cPUCT direction grid:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_logcpuct_d5_black_20260514_105131_setsid
PID: 207123
```

Configuration:

```text
opening_suite = arena_openings/human_val_opening_suite_v1.json
max_openings = 4
games_per_opening = 1
opponent = Pikafish d5
AlphaXiang side = black
sims = 8000
c_puct = 1.25, 1.30
c_puct_base = 19652
c_puct_factor = 0.5, 0.75
q_weight = 1.0
fpu_reduction_root/tree = legacy disabled (-1.0)
temperature_move = 0.02
```

Purpose: test whether a lower initial exploration value with visit-scaled growth
can outperform fixed `c_puct=1.45` on the same Pika d5 black-side pressure cell.

Result:

```text
summary = /home/laure/alphaxiang/v14s_search_tuning/phase1_logcpuct_d5_black_20260514_105131_setsid/v14s_search_grid_summary.md
```

| c_puct | c_puct_base | c_puct_factor | W-L-D | score | terminations |
|---:|---:|---:|---:|---:|---|
| 1.25 | 19652 | 0.50 | 1-2-1 | 37.5% | max 1, longcheck 1, mate 2 |
| 1.25 | 19652 | 0.75 | 0-3-1 | 12.5% | mate 3, nocap 1 |
| 1.30 | 19652 | 0.50 | 0-2-2 | 25.0% | nocap 2, mate 2 |
| 1.30 | 19652 | 0.75 | 0-4-0 | 0.0% | mate 4 |

Auto-continuation decision:

```text
best score = 37.5% < 50.0% threshold
FPU follow-up was not launched
```

Interpretation: this log-cPUCT shape is directionally worse than fixed
`c_puct=1.45`, which scored 50.0% over the larger 12-opening validation.  The
larger `factor=0.75` cells are especially bad and increase mate losses.  Keep
the implementation available for later, but do not promote this schedule.

## Interpretation Rules

- Treat 2-game cells as noisy directional hints only.
- Prefer configs that reduce mate losses, not merely configs that draw more by
  max-plies or no-capture.
- If `c_puct` sensitivity is large, run a second grid with:

```text
c_puct around the best value +/- 0.15
q_weight = 0.85, 1.0, 1.15
temperature_move = 0.0, 0.02
```

- If no `c_puct` setting beats baseline directionally, move to q-weight/q-clip
  and then consider C++ changes such as log-scaled cPUCT or root/tree FPU.

## Guardrails

- Do not use Gumbel root selection as final move selection.
- Do not train V14 adapters during this phase.
- Do not overwrite V13 or V14 assets.
- All outputs must stay under `/home/laure/alphaxiang/v14s_search_tuning`.

## First Coarse Grid Result

Finished:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_coarse_d5_black_20260514_010155_setsid
```

Results, Pika d5, AlphaXiang black-only, 8000 sims, 2 games per cell:

| c_puct | W-L-D | score | termination pattern |
|---:|---:|---:|---|
| 1.00 | 0-2-0 | 0.0% | 2 mate losses |
| 1.25 | 0-1-1 | 25.0% | 1 mate loss, 1 no-capture draw |
| 1.50 | 0-0-2 | 50.0% | 2 no-capture draws |

Early interpretation: higher `c_puct` looks directionally better for black-side d5 defense.  This is only a 2-game smoke, but it is consistent with the hypothesis that V13 needs more exploration support at high sims to avoid defensive tactical collapse.

## Second Focused Expansion

Started:

```text
/home/laure/alphaxiang/v14s_search_tuning/phase1_expand_cpuct_d5_black_20260514_032531_setsid
PID: 200153
```

Configuration:

```text
opponent = Pikafish depth 5
AlphaXiang side = black
sims = 8000
games per combo = 4
temperature_move = 0.02
q_weight = 1.0
q_clip = 1.0
c_puct = 1.4, 1.5, 1.6
```
