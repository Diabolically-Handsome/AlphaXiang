# V14V Top-K Pikafish Verifier

Date: 2026-05-14

## Goal

Do not change model weights.  Keep V13 as the main player, then add a
conservative alpha-beta style verifier at root:

```text
V13 MCTS chooses a move from the root
take root top-K candidate moves
apply each candidate move
ask Pikafish to evaluate the child position
override V13 only if another candidate is safer by a large cp margin
```

This tests the core hypothesis from the Compass plan: V13 needs tactical search
behind its global policy, not another static CNN head.

## Implementation

Changed:

```text
tools/pikafish_opponent.py
tools/external_arena.py
```

New Pikafish wrapper APIs:

```text
go_depth_eval()
go_nodes_eval()
go_movetime_eval()
```

These return:

```text
bestmove, ponder, score_cp, mate_in
```

Score convention: Pikafish cp is from side-to-move perspective.  After our
candidate root move, the child side-to-move is the opponent, so lower child cp is
better for AlphaXiang.

New arena flags:

```text
--our-pikafish-verifier
--our-verifier-top-k
--our-verifier-margin-cp
--our-verifier-danger-threshold-cp
--our-verifier-depth
--our-verifier-nodes
--our-verifier-movetime-ms
--our-verifier-threads
--our-verifier-hash-mb
--our-verifier-side any|red|black
```

Default behavior is unchanged because the verifier is disabled unless
`--our-pikafish-verifier` is set.

## Safety

The verifier is conservative:

```text
replacement only if original_child_eval_cp - candidate_child_eval_cp >= margin_cp
```

Prototype margin:

```text
margin_cp = 120
```

High-risk gating:

```text
danger_threshold_cp = -20000  # old behavior / no gate
danger_threshold_cp = 600     # only override when the original move is already
                              # clearly dangerous from the opponent POV
```

Only actual replacements are logged as guard events, so JSON output does not
explode from every checked root move.

## Smoke Tests

Passed:

```text
python -m py_compile tools/pikafish_opponent.py tools/external_arena.py
Pikafish go_depth_eval depth=1 from startpos
external_arena random-opponent interface smoke with verifier enabled
```

Interface smoke output:

```text
/home/laure/alphaxiang/v14v_pikaverifier/interface_smoke/external_arena_20260514_224420.json
```

## First Arena Probe

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d4_20260514_224549_setsid
PID: 212533
```

Configuration:

```text
checkpoint = run031a step19000
opponent = Pikafish d5
AlphaXiang side = black
opening_suite = arena_openings/human_val_opening_suite_v1.json
max_openings = 4
games_per_opening = 1
our_sims = 8000
c_puct = 1.45
q_weight = 1.0
temperature_move = 0.02
ship safety = root mate1 guard + mate1/mate2 extension
verifier_top_k = 3
verifier_depth = 4
verifier_margin_cp = 120
verifier_side = black
```

Decision target:

```text
Compare against fixed c_puct=1.45 4-opening smoke: 3W-1L-0D
Also compare directionally against fixed c_puct=1.45 12-opening validation: 5W-5L-2D
```

Interpretation rule:

If the verifier reduces mate/longcheck losses without causing many bad
overrides, continue to deeper verifier or larger 12-opening validation.  If it
underperforms fixed `c_puct=1.45`, do not move directly to NNUE yet; first review
override events to see whether the cp sign/margin/gating needs adjustment.

## Depth-4 Result

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d4_20260514_224549_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d4_20260514_224549_setsid/external_arena_20260514_231940.json
```

Result:

```text
1W-2L-1D / 4 = 37.5%
terminations = nocap 1, mate 3
verifier override events = 13
games_with_events = 4
```

Interpretation: depth-4 verifier underperformed fixed `c_puct=1.45` and
overrode too often.  User hypothesis: the verifier teacher may simply be too
weak, because current V13 can already compete with low-depth Pikafish.  Next
probe keeps the same top-K/margin shape but raises the verifier teacher to
depth 10.

## Depth-10 Teacher Probe

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_20260515_061255_setsid
PID: 214035
```

Configuration:

```text
checkpoint = run031a step19000
opponent = Pikafish d5
AlphaXiang side = black
opening_suite = arena_openings/human_val_opening_suite_v1.json
max_openings = 4
games_per_opening = 1
our_sims = 8000
c_puct = 1.45
q_weight = 1.0
temperature_move = 0.02
ship safety = root mate1 guard + mate1/mate2 extension
verifier_top_k = 3
verifier_depth = 10
verifier_margin_cp = 120
verifier_side = black
```

Purpose: isolate the "teacher too weak" hypothesis by changing verifier depth
from 4 to 10 while keeping the override policy mostly unchanged.

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_20260515_061255_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_20260515_061255_setsid/external_arena_20260515_064109.json
```

Result:

```text
1W-1L-2D / 4 = 50.0%
terminations = nocap 2, mate 2
guard/verifier events = 8
root_pikafish_topk_verifier events = 7
root_mate1_blunder_guard events = 1
games_with_events = 4
```

Interpretation: raising the verifier teacher from depth 4 to depth 10 improved
the result from 37.5% to 50.0%, so the "teacher too weak" hypothesis is
directionally supported.  However, the result only matched the fixed
`c_puct=1.45` 12-opening validation and remained below the fixed 4-opening
smoke result.  It also still replaced moves often, with 7 verifier overrides in
4 games.  The next probe should keep depth 10 but raise the override margin so
Pikafish acts as a high-confidence veto rather than a frequent co-player.

## Depth-10 Conservative Margin Probe

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_m300_20260515_075545_setsid
PID: 215992
```

Configuration difference from the previous depth-10 probe:

```text
verifier_margin_cp = 300
seed = 2026051530
```

Purpose: test whether a stronger teacher helps when it is allowed to override
only clear tactical refutations.  If this reduces override count while staying
at or above 50%, it is a better candidate for 12-opening validation than the
margin-120 verifier.

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_m300_20260515_075545_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase1_d5_black_top3_d10_m300_20260515_075545_setsid/external_arena_20260515_081647.json
```

Result:

```text
3W-1L-0D / 4 = 75.0%
terminations = mate 4
guard/verifier events = 0
root_pikafish_topk_verifier events = 0
games_with_events = 0
```

Interpretation: margin 300 matched the fixed `c_puct=1.45` 4-opening smoke
score while avoiding the over-override behavior seen at margin 120.  This does
not yet prove the verifier improves strength, because no verifier replacement
actually fired in this 4-game sample.  It does prove that conservative gating is
not harmful on this small set.  Next step: run a 12-opening validation at
depth 10, margin 300.  If event count stays near zero, lower the margin to 220
or add high-risk-only triggering rather than moving to margin 500.

## Depth-10 Margin-300 12-Opening Validation

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_20260516_022648_setsid
PID: 218644
```

Configuration difference from the 4-opening margin-300 smoke:

```text
max_openings = 12
seed = 2026051601
```

Purpose: compare the conservative depth-10 verifier against the fixed
`c_puct=1.45` 12-opening validation baseline of `5W-5L-2D = 50.0%`.

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_20260516_022648_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_20260516_022648_setsid/external_arena_20260516_033634.json
```

Result:

```text
6W-5L-1D / 12 = 54.2%
terminations = mate 9, nocap 1, longcheck 2
guard/verifier events = 5
root_pikafish_topk_verifier events = 3
root_mate1_blunder_guard events = 2
games_with_events = 3
```

Interpretation: this is a modest improvement over the fixed `c_puct=1.45`
12-opening baseline, but not large enough to call a breakthrough.  Margin 300
did what we wanted structurally: it triggered occasionally rather than taking
over the player.  Run a second seed on the same 12-opening suite before deciding
whether margin 300 is a release candidate or only noise.

## Depth-10 Margin-300 12-Opening Validation Seed 2

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_seed2_20260516_040236_setsid
PID: 219734
```

Configuration difference from seed 1:

```text
seed = 2026051602
```

Purpose: test whether the seed-1 improvement persists across the same
12-opening suite with different stochastic move sampling.

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_seed2_20260516_040236_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase2_d5_black_top3_d10_m300_12open_seed2_20260516_040236_setsid/external_arena_20260516_053031.json
```

Result:

```text
4W-6L-2D / 12 = 41.7%
terminations = mate 8, rep 1, nocap 1, longcheck 2
guard/verifier events = 9
root_pikafish_topk_verifier events = 8
root_mate1_blunder_guard events = 1
games_with_events = 3
```

Combined result with seed 1:

```text
10W-11L-3D / 24 = 47.9%
```

Interpretation: ungated margin 300 is not a release candidate.  Seed 2 had
more verifier overrides and a worse score, so the verifier is still too willing
to intervene.  The next shape should preserve the strong teacher but only allow
overrides when the originally selected move is already dangerous.

## Depth-10 Margin-300 High-Risk Gate

Code change:

```text
tools/external_arena.py
```

Added:

```text
--our-verifier-danger-threshold-cp
```

Default `-20000` preserves old behavior.  The first high-risk probe uses `600`.

Started:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase3_d5_black_top3_d10_m300_danger600_12open_seed2_20260516_055223_setsid
PID: 221470
```

Configuration:

```text
verifier_depth = 10
verifier_margin_cp = 300
verifier_danger_threshold_cp = 600
seed = 2026051602
max_openings = 12
```

Purpose: direct A/B against the failed ungated seed 2.  This keeps terminal and
large tactical danger overrides, while suppressing low-confidence
300-400cp-style interventions that may be disrupting the model's own search.

Completed:

```text
/home/laure/alphaxiang/v14v_pikaverifier/phase3_d5_black_top3_d10_m300_danger600_12open_seed2_20260516_055223_setsid
json = /home/laure/alphaxiang/v14v_pikaverifier/phase3_d5_black_top3_d10_m300_danger600_12open_seed2_20260516_055223_setsid/external_arena_20260516_071225.json
```

Result:

```text
4W-5L-3D / 12 = 45.8%
terminations = mate 7, rep 1, nocap 2, longcheck 2
guard/verifier events = 1
root_pikafish_topk_verifier events = 1
games_with_events = 1
```

Only replacement event:

```text
game 10, ply 133, black:
original e8d9 -> replacement e8d7
original child eval = +20000cp opponent POV, mate_in=5
replacement child eval = +825cp opponent POV
final game result = opp_win
```

Interpretation: the high-risk gate reduced verifier interventions from 8 events
in the failed ungated seed-2 run to 1 event, but it did not beat the fixed
`c_puct=1.45` baseline.  The single override correctly identified a terminal
candidate as dangerous, yet the replacement still left AlphaXiang in a poor
position and the game was lost.  This means `danger_threshold_cp=600` is safer
than ungated margin-300 behavior, but it is not a release candidate.  The next
step is event-level auditing plus paired same-seed A/B against baseline, not
another blind verifier-parameter sweep.
