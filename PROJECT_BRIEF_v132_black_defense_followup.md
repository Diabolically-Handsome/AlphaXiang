# V13.2 Black Defense Follow-up

## Summary

- Confirmed the Pika d5 losses under `mate1+mate2` were entirely an AlphaXiang-as-black problem.
- Added black-only arena support and two root defensive guard probes.
- Exact mate2 / forcing-check root guards did not yet explain all black losses, because they rarely triggered.
- Created a black-only d5 loss slice for the next training-side repair.

## Confirmed Color Split

For `V13.2 mate1+mate2` Pika d5, 20 games:

| side | W-L-D | score |
|---|---:|---:|
| AlphaXiang red | 8-0-2 / 10 | 90.0% |
| AlphaXiang black | 0-6-4 / 10 | 20.0% |
| aggregate | 8-6-6 / 20 | 55.0% |

Conclusion:

- Aggregate score hides the bottleneck.
- The current system is much stronger as the attacking/first-moving side.
- Black-side defensive king-safety remains the main d5 weakness.

## Code Changes

Changed:

- `tools/external_arena.py`
- `tools/arena_failure_slice.py`

New arena flags:

```bash
--our-side alternate|red|black
--our-root-mate2-blunder-guard
--our-root-forcing-check-guard-plies N
--our-root-forcing-check-guard-max-candidates K
```

New failure-slice flag:

```bash
--our-side-filter any|red|black
```

## Black-only Tests

Base:

```text
022d step18000
8000 sims
Pika d5
our-side black
mate1 + mate2 leaf extension
root mate1 + root mate2 guards
```

Result:

```text
0W-3L-3D / 6
score = 25.0%
root guard events = 0
```

Interpretation:

- Root mate2 guard did not fire.
- Black losses are probably not simple "black move allows red check-forced mate2".

Broader root forcing-check guard:

```text
--our-root-forcing-check-guard-plies 5
```

Result:

```text
2W-1L-1D / 4
score = 62.5%
root guard events = 0
```

Interpretation:

- Result is promising but not causal, because the guard did not actually replace moves.
- Treat as small-sample variance / leaf mate2 effect, not proof that forcing5 root guard fixed black.

Even broader:

```text
--our-root-forcing-check-guard-plies 7
```

Result:

```text
0W-0L-2D / 2
score = 50.0%
root guard events = 0
```

Interpretation:

- Very slow.
- Still no root events.
- Exact root forcing-check detection is probably too narrow, or the losing mechanism is longer / more positional than check-only mate2/3.

## Black Loss Slice

Created:

```text
/home/laure/alphaxiang/v132_black_d5_loss_slice_raw
```

Manifest:

```text
games_seen = 70
games_used = 19
samples = 1247
our_side_filter = black
only_our_turns = true
```

This is a raw black-defense failure slice. It still needs teacher relabeling before training.

Recommended labeling:

```bash
oracle_policy_labeler.py depth=8 multipv=5 adaptive-temperature legal-smoothing=0.05
action_value_labeler.py depth=12 include-chosen model-top-k
```

## Current Diagnosis

The black weakness is real, but not fully explained by exact mate2 at root.

Most likely remaining causes:

1. Red has longer forcing-check pressure than mate2/3.
2. Black king-safety defense requires quiet defensive moves, not just avoiding immediate forced mate.
3. Leaf mate2 improves attack more than defense.
4. Training-side repair should target black defensive positions with pairwise good-vs-bad labels.

## Recommendation

- Keep `mate1+mate2` as V13.2 experimental high-strength / data-generation mode.
- Do not claim root mate2/forcing guard fixed black yet.
- Next real fix should be:
  1. label `/home/laure/alphaxiang/v132_black_d5_loss_slice_raw`;
  2. construct `good_move > bad_move` black defensive pairs;
  3. train a tiny anchor-KL pairwise repair from 022d;
  4. evaluate with mandatory red/black split.
