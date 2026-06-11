# V13.2 Mate2 / Forcing-Check Safety Smoke Results

## Summary

- Implemented a new optional MCTS leaf probe: `--our-tactical-mate2-extension`.
- It is deliberately conservative:
  - only detects check-first forced mate-in-2;
  - ignores quiet mate nets;
  - does not change model weights;
  - does not replace the existing `022d step18000` checkpoint.
- Early results are stronger than mate1-only safety, but runtime is noticeably slower.

## Code Changes

Files changed:

- `xqcpp_ext_hist8_115.cpp`
- `xiangqi_mcts_ext.py`
- `tools/external_arena.py`

New CLI flag:

```bash
--our-tactical-mate2-extension
```

Recommended test preset:

```bash
--our-root-mate1-blunder-guard \
--our-tactical-mate1-extension \
--our-tactical-mate2-extension
```

## Mechanism

At a neural leaf, before asking the model value head, MCTS now checks:

1. Does side-to-move already have mate-in-1?
2. If not, does side-to-move have a checking move such that every legal reply still allows mate-in-1?

If yes, the leaf value is backed up as `+1.0`.

This is a neuro-symbolic search correction, not a training change.

## Results

Base checkpoint:

```text
/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500/snapshots/latest_step18000.pt
```

Search:

```text
8000 sims, q_weight=1.0, temperature_move=0.02
```

| anchor | games | W-L-D | score |
|---|---:|---:|---:|
| Pika d3 | 6 | 4-0-2 | 83.3% |
| Pika d4 | 6 | 3-0-3 | 75.0% |
| Pika d5 | 20 | 8-6-6 | 55.0% |

Evidence files:

- `/home/laure/alphaxiang/v132_mate2_forcing_safety_smoke/pika_d3_mate1_mate2_guard/external_arena_20260510_005758.json`
- `/home/laure/alphaxiang/v132_mate2_forcing_safety_smoke/pika_d4_mate1_mate2_guard/external_arena_20260510_011943.json`
- `/home/laure/alphaxiang/v132_mate2_forcing_safety_smoke/pika_d5_mate1_mate2_guard/external_arena_20260509_230549.json`
- `/home/laure/alphaxiang/v132_mate2_forcing_safety_expand20/pika_d5_mate1_mate2_guard/external_arena_20260510_002423.json`

Comparison against V13.1 mate1-only safety:

| config | Pika d5 games | score |
|---|---:|---:|
| mate1-only safety | 100 | 47.5% |
| mate1 + mate2 safety | 20 | 55.0% |

## Interpretation

This is a promising result. Mate2 safety appears to improve the exact weakness exposed by Pika d5: forcing tactical refutations near king safety.

The caveat is speed. Mate2 is much slower than mate1-only because each leaf may need a check-first forced-mate probe. It should not be treated as a PVP default yet.

Best current use:

1. High-strength analysis mode.
2. Data generation for V13.2 tactical curriculum.
3. Failure-labeling of d5 losses.

## Recommendation

- Do not replace V13.1 mate1-only safety as the normal default yet.
- Promote mate2 safety to `V13.2 experimental high-strength / data-generation mode`.
- Next validation:
  - expand Pika d5 mate2 to 50 or 100 games;
  - split red/black results;
  - profile runtime cost;
  - generate tactical refutation shards from remaining mate2 losses.
