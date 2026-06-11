# PROJECT BRIEF: v13 High-Sims Search-Budget Unlock

## Summary
- v13 `run_022c` step1500 has a clear search-budget unlock pattern: low sims underestimates strength, while high sims reveals much stronger play.
- The current practical sweet spot on Pika d3 is around `8000` sims: it matches `12800` score in 50 games while using much less wall-clock time.
- `12800` sims is still useful as a stability probe: it produced zero losses in the 50-game Pika d3 run, but did not improve score over `8000`.
- This supports the paper hypothesis cautiously: larger Transformer models may require larger MCTS budgets to express their policy/value quality. This should later be tested under both fixed-sims and fixed-wall-clock settings.

## Verified Results

Base checkpoint:

`/home/laure/alphaxiang/training_runs/run_022c_v13_nopool_widened_mild_teacherq_from022a1000/snapshots/latest_step1500.pt`

| Opponent | Sims | Games | Result | Score | Avg plies | JSON |
|---|---:|---:|---:|---:|---:|---|
| Pika d3 | 6400 | 50 | 23W-8L-19D | 65.0% | 119.12 | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022c_recheck_sims6400_50g/step1500/pika_d3/external_arena_20260505_140420.json` |
| Pika d3 | 8000 | 50 | 35W-2L-13D | 83.0% | 105.84 | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022c_recheck_sims8000_50g/step1500/pika_d3/external_arena_20260505_161241.json` |
| Pika d3 | 12800 | 50 | 33W-0L-17D | 83.0% | 111.38 | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022c_recheck_sims12800_50g/step1500/pika_d3/external_arena_20260505_195127.json` |
| Pika d4 | 6400 | 20 | 14W-2L-4D | 80.0% | 109.75 | `/home/laure/alphaxiang/v13_snapshot_smoke/run_022c_recheck_sims6400_20g/step1500/pika_d4/external_arena_20260505_125147.json` |

Earlier reference points:

- Pika d3 @ 800 sims, 20 games: `4W-10L-6D`, 35.0%.
- Pika d3 @ 1600 sims, 50 games: `13W-26L-11D`, 37.0%.
- Pika d3 @ 3200 sims, 20 games: `11W-2L-7D`, 72.5%.

## Interpretation
- The jump from `1600` to `3200+` sims is large enough that v13 should not be judged by the v12-era default `1600` sims alone.
- The jump from `6400` to `8000` is also substantial in this sample: `65.0% -> 83.0%`.
- The jump from `8000` to `12800` does not improve score in 50 games, but reduces losses from `2` to `0`.
- Current operational default for expensive v13 anchor testing should be `8000` sims. Use `12800` only for decisive stability probes or final paper-grade confirmation.

## Caveats
- These are still small panels, especially the Pika d4 `6400` result with only 20 games.
- The conclusion should not be phrased as a universal law yet. A fair paper ablation should compare:
  - fixed sims across model sizes;
  - fixed wall-clock across model sizes;
  - score per unit compute;
  - at least v12.6-micro, v13 022c/022d, and the CNN baseline.
- Pika d3 saturation at high sims does not automatically prove broad generalization. Pika d4/d5, Fairy, and CNN anchors still matter.

## Current Follow-Up
- Overnight training is running as:

`/home/laure/alphaxiang/training_runs/run_022d_v13_nopool_widened_overnight_from022c1500`

- Start checkpoint:

`/home/laure/alphaxiang/training_runs/run_022c_v13_nopool_widened_mild_teacherq_from022a1000/snapshots/latest_step1500.pt`

- Initial checkpoints observed:
  - `latest_step2000.pt`: human validation total loss `3.0679`
  - `latest_step3000.pt`: human validation total loss `3.0555`
  - `latest_step4000.pt`: human validation total loss `3.0669`

Current judgement: stable continuation, no stop condition triggered.
