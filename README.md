# AlphaXiang Transformer

A Transformer-based AlphaZero-style engine for **Xiangqi** (Chinese Chess), trained with MCTS self-play and Pikafish distillation.

This repo covers **Stage 1** of the project: bootstrapping the engine from random init
to a reasonable playing strength using a mix of human-game supervised pretraining,
Pikafish-labelled distillation, adversarial self-play vs Pikafish, and MCTS
self-play against past snapshots.

---

## Model

- 12-layer Transformer, d_model=512, 8 heads, FFN 2048 → ~140M params
- Input: 115-plane 10×9 board representation (8 history snapshots + side/clock planes)
- Three heads:
  - **Policy** — from×to logits over 8100 moves (90 squares × 90 squares)
  - **Value (scalar)** — tanh-bounded win-expectation, MSE-trained
  - **WDL (3-class)** — Win/Draw/Loss probabilities, cross-entropy-trained

See `xiangqi_transformer_model.py` for the architecture.

---

## Stage 1 outcome

Starting from a randomly-initialized transformer, Stage 1 trained to
**step 181,000** over ~3 weeks. External-arena calibration against Pikafish at
fixed depth gives an approximate Elo of **~1387** for the final checkpoint
(`best.pt`), measured as:

| Cycle | sims / depth | Games | W-L-D | Score rate |
|-------|--------------|------:|:-----:|-----------:|
| c1    | 800 / 8      |    50 | 11-38-1 | 22.5% |
| c2    | 800 / 8      |    50 |  7-42-1 | 15.0% |
| c3    | 800 / 8      |    50 | 10-40-0 | 20.0% |
| c4    | 800 / 8      |    50 | 11-36-3 | 25.0% |
| c5    | 800 / 8      |    50 | 12-35-3 | 27.0% |
| c6    | 800 / 8      |    50 | 10-39-1 | 21.0% |
| c7    | 800 / 8      |    50 | 13-35-2 | 28.0% |
| **Aggregate** | | **350** | **74-265-11** | **22.7%** |

The engine is not yet competitive with strong engines; the next stage will
raise it using a self-play curriculum with stronger adversarial partners.

---

## Repo layout

```
xiangqi_transformer_model.py    Model definition (policy + scalar + WDL heads)
xiangqi_train.py                Training loop, replay buffer, shard ingest, arena trigger
xiangqi_selfplay.py             MCTS self-play worker (writes tensorized shards)
xiangqi_arena.py                Candidate-vs-best match harness + acceptance rules
xiangqi_closed_loop.py          Orchestrator: alternates selfplay / train / arena
xiangqi_mcts_ext.py             Python bindings for the C++ MCTS/Board extension
xqcpp_ext_hist8_115.cpp         C++ MCTS / board / move-gen / 115-plane encoder
xiangqi_model_battle_gui.py     Tkinter GUI to play against a checkpoint

tools/
  distillation_generator.py     Random rollouts → Pikafish labels → tensorized shards
  pikafish_pool.py              Multiprocess Pikafish worker pool for batch labelling
  pikafish_opponent.py          Single-Pikafish UCI wrapper (used by vs-Pikafish self-play)
  pikafish_selfplay.py          Our MCTS vs Pikafish with Tier-1 noise (adversarial data)
  stage1_driver.py              Distill-then-vsPikafish-then-train loop for Stage 1
  external_arena.py             Calibration harness (our engine vs Pikafish at fixed depth)
  marathon_watchdog.sh          Long-run supervisor (detects stalls, restores from best.pt)
  verify_checkpoints.py         Quick integrity check over checkpoint set
  simulate_arena_logic.py       Offline simulator for the arena acceptance rules
  summarize_arenas.py           Arena result rollup
  today_arenas.py               Lists today's arena runs

CNN/                            Earlier CNN-based baseline (pre-Transformer)
```

Data directories (`human_bootstrap_data*`, `selfplay_runs*`, `training_runs/`,
`arena_runs/`) are produced by running the pipeline and are not checked in.

---

## Key Stage 1 changes (vs the initial plan)

The initial transformer bootstrap got stuck in a ~60% draw loop. Root-cause
analysis identified a **frozen WDL head** collapsing the value signal, plus a
deterministic arena that generated repetition draws. The fixes that unblocked
training:

1. **Unfreeze the WDL head** and train it with 3-class cross-entropy
   (`wdl_loss_weight=1.0`). Scalar-value MSE kept at `value_loss_weight=0.5`
   and draw targets scaled to `±0.9` (`value_target_scale`) so draws don't
   pull W/L samples towards zero.

2. **Arena root noise on by default** (`add_root_noise=True`) plus move
   temperature 0.5 — avoids the deterministic repetition traps that
   made candidates indistinguishable from the champion.

3. **Dual acceptance criterion** in the arena: accept on either
   `winrate ≥ 0.55 AND non_draw ≥ 10`, *or*
   `decisive_winrate ≥ 0.60 AND decisive_total ≥ 10`. This unblocks promotion
   when draw rate is high.

4. **Tighter draw prevention**: `no_capture_limit: 96→60` (traditional
   Xiangqi 60-half-move rule), plus earlier capture-priority thresholds
   in self-play.

5. **Pikafish distillation + adversarial curriculum** (`tools/stage1_driver.py`):
   each cycle generates 30% distillation shards (random-rollout positions
   labelled by Pikafish at depth 6) and 70% vs-Pikafish shards (our MCTS vs
   Pikafish depth 3 with 15% random noise injected against us). This gives
   the value and WDL heads a dense, engine-grade label distribution.

See `xiangqi_train.py` (TrainingConfig, `_compute_training_losses`),
`xiangqi_arena.py` (acceptance rule around line 750),
and `tools/stage1_driver.py` for the full implementation.

---

## Dependencies

- Python 3.10+
- PyTorch with CUDA (trained on RTX 5090)
- [Pikafish](https://github.com/official-pikafish/Pikafish) binary for distillation and calibration
- The C++ extension in `xqcpp_ext_hist8_115.cpp` requires a matching
  pybind11 build (not automated here; see the extension source for build flags)

---

## Dual-GPU mode (Stage 2 prep)

`tools/stage1_driver.py` supports running the vs-Pikafish self-play phase and
the training phase IN PARALLEL on two different CUDA devices. This gives a
meaningful wall-clock reduction per cycle when you have asymmetric GPUs where
one card is VRAM-rich (good for training) and the other has less VRAM but is
still strong for inference (good for MCTS rollouts).

Example (RTX 5090 training + RTX 5080 self-play):

```bash
python tools/stage1_driver.py \
    --cycles 7 \
    --train-device cuda:0 \
    --selfplay-device cuda:1 \
    [... other args ...]
```

When `--train-device` and `--selfplay-device` are **different**, the driver:

1. Runs distillation (CPU-only; Pikafish NNUE workers) — same as before
2. Launches vspika (GPU 1) and training (GPU 0) as **parallel subprocesses**
3. Kills the sibling if either fails, so no orphan GPU processes
4. Emits a heartbeat line every 60s so log-stall watchdogs don't false-alarm

If both devices are the same (or only `--device cuda:0` is given), the driver
falls back to the legacy serial path. No behaviour change for single-GPU users.

Note: in parallel mode, the current cycle's vspika shards become visible to
training only on the **next** cycle (vspika writes its manifest.json at the
end). Distill shards and prior-cycle vspika shards are still available to
training during the current cycle. This is the intended trade-off — full
continuous-pipeline mode (where training reads shards as they're written
mid-cycle) is a Stage 2 upgrade.

---

## Status

Stage 1 is complete. The final checkpoint is `best.pt` at step 181,000.
Stage 2 (stronger curriculum, Fairy-Stockfish, possibly a larger model,
dual-GPU parallel pipeline) is in progress — dual-GPU support landed in
`tools/stage1_driver.py` as the opening change.
