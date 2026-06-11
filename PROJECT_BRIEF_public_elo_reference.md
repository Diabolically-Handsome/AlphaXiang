# AlphaXiang Public Elo Reference

## Summary

For public-course slides, use:

> AlphaXiang V13.3 is approximately **2400 public-engine-anchor Elo**
> under our fixed Xiangqi engine benchmark, with a conservative range of
> **2350-2450**.

This is **not an official human federation Elo**. It is a public-facing reference
number derived from reproducible engine anchors. The safest wording is:

> "about 2400 strength on our public Xiangqi engine-anchor ladder"

rather than:

> "officially 2400 human Elo"

## Current Best Candidate

- Checkpoint:
  `/home/laure/alphaxiang/training_runs/run_031a_v133_p6_fullpika_black_blunder_round2_from030a18500/snapshots/latest_step19000.pt`
- Recommended inference config:
  - `our_sims=8000`
  - `q_weight=1.0`
  - `temperature_move=0.02`
  - `root_mate1_blunder_guard=on`
  - `tactical_mate1_extension=on`
  - `tactical_mate2_extension=on`

## Evidence Behind The 2400 Number

Known earlier anchor:

- v12.6-micro was estimated around **2080-2150**.
- v13 `022d step18000` moved into roughly **2300-2400** by public-engine anchors:
  - Pika d3: `62.0%`
  - Pika d4: `60.0%`
  - Pika d5: `46.0%`
  - Fairy d3: `94.0%`
  - CNN best: `92.0%`

Current candidate:

- `run031a step19000` was selected over later repair checkpoints because it kept the
  best overall tradeoff.
- It improved the previously weak black-side d5 behavior:
  - Pika d5 black-only quick check: `2W-0L-4D / 6 = 66.7%`
  - Pika d6 black-only quick check: `1W-4L-1D / 6 = 25.0%`
  - Pika d6 red-only 20-game check: `1W-9L-10D / 20 = 30.0%`

Interpretation:

- AlphaXiang is clearly above the v12.6 2100-ish range.
- It is strong enough to challenge Pika d5 in some settings, but Pika d6 still exposes
  high-pressure tactical and conversion weaknesses.
- Therefore **2350-2450** is the honest range; **2400** is the clean single-number
  public reference.

## Why We Are Not Using Pikafish UCI_Elo Yet

The ideal method would be to run a direct ladder against Pikafish with:

- `UCI_LimitStrength=true`
- `UCI_Elo=2200/2400/2600/...`

However, both the local Pikafish binary and the official Linux binary from
`Pikafish 2026-01-02` do **not** expose `UCI_LimitStrength` or `UCI_Elo` in their
actual `uci` option list.

I added support for this path, but the runner now refuses to run if those UCI options
are missing, because otherwise a supposed "Pika Elo 2200" test is actually just
full-strength Pikafish at fixed movetime.

Implemented files:

- `tools/external_arena.py`
  - added `--opp-uci-elo`
  - added `--opp-uci-limit-strength`
- `tools/public_elo_ladder.py`
  - runs a Pikafish UCI_Elo ladder when supported
  - validates that the selected binary exposes `UCI_LimitStrength` and `UCI_Elo`
  - writes `public_elo_summary.json` and `public_elo_summary.md`

Validation result:

- Official `Pikafish 2026-01-02` Linux binary exposes:
  `Threads`, `Hash`, `MultiPV`, `Move Overhead`, `UCI_ShowWDL`, `EvalFile`, etc.
- It does **not** expose:
  `UCI_LimitStrength`, `UCI_Elo`.

## Public Sources

- Pikafish is an official open-source UCI Xiangqi engine:
  https://github.com/official-pikafish/Pikafish
- Official Pikafish release checked:
  https://github.com/official-pikafish/Pikafish/releases
- Pikafish Wiki documents UCI options, but actual Linux binaries checked here do not
  currently expose `UCI_Elo`:
  https://www.pikafish.com/wiki/index.php?title=UCI%E9%80%89%E9%A1%B9
- Langer 2021 evaluates Xiangqi neural engines against Fairy-Stockfish and public
  engines, and describes Fairy-Stockfish as master-level:
  https://ml-research.github.io/papers/langer2021xiangqi.pdf
- Xiangqi.com explicitly warns that its ratings are a closed-pool relative rating,
  not an absolute measure of Xiangqi skill:
  https://www.xiangqi.com/help/rating

## Recommended Public-Course Wording

Short version:

> AlphaXiang V13.3 is about **2400 Elo on a public Xiangqi engine-anchor scale**.

Safer full version:

> AlphaXiang V13.3 is not rated in an official human federation pool. Against our
> reproducible public Xiangqi engine anchors, its current strength is best summarized
> as **about 2400 engine-anchor Elo**, with a conservative uncertainty band of
> **2350-2450**. It is strong enough to beat most casual and club-level players, but
> still has tactical and endgame-conversion weaknesses against high-depth Pikafish.

