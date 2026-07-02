# AlphaXiang

[![CI](https://github.com/Diabolically-Handsome/AlphaXiang/actions/workflows/ci.yml/badge.svg)](https://github.com/Diabolically-Handsome/AlphaXiang/actions/workflows/ci.yml)

A Transformer-based, AlphaZero-style **Xiangqi (Chinese Chess)** engine — distilled from
a deep search oracle, then improved by self-play with a component-isolated training loop.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # perft + rules + model spec (~1 min incl. C++ JIT compile)
pytest -q -m slow         # deep perft (3.29M nodes)
```

The first import JIT-compiles the C++ MCTS extension (needs a C++17 toolchain; ninja
comes from requirements.txt). To play against a trained network, download a checkpoint
from the [releases](https://github.com/Diabolically-Handsome/AlphaXiang/releases),
`pip install pygame`, and run `xiangqi_model_battle_gui.py` with `--transformer-checkpoint`.

## Correctness

- **Perft-verified move generator**: the C++ movegen reproduces the community-standard
  Xiangqi perft counts from the starting position — 44 / 1,920 / 79,666 / 3,290,240
  at depths 1–4 ([tests/test_perft.py](tests/test_perft.py))
- The optimized in-check routine is cross-checked against a generator-based slow
  reference across seeded random playouts ([tests/test_board.py](tests/test_board.py))
- Model spec pinned by tests: exact parameter counts (38,641,766 with both attention
  biases), head shapes, finite gradients ([tests/test_model.py](tests/test_model.py))

## Model

- **38.6M parameters**: 12-layer Transformer, `d_model=512`, 8 heads, FFN 2048
- **Geometry-aware attention**: 2D relative-position bias + line-of-sight bias
- Input: 115 planes over the 10×9 board (8 history snapshots + side/clock planes)
- Three heads: policy (from×to logits over 8100 moves), scalar value (tanh), WDL (3-class)
- Initial strength via supervised distillation from **Pikafish at search depth 20**

## The interesting part: frozen-evaluator self-play (v16)

Naive AlphaZero-style self-play *regressed* our distilled model at every gate — until
component-level diagnosis (see below) showed the **policy had been improving all along**
while noisy outcome targets silently degraded the **value head**, and the package-level
gate could only see the sum. The current training loop therefore separates the two:

- **Self-play and gating run policy/value *chimeras***: the candidate supplies the
  policy, a **frozen reference network supplies the value** — search always runs on a
  healthy, calibrated evaluator while only the policy learns.
- Training is **policy-only** (value/WDL loss weights 0), with the optimizer reset on
  resume so the configured learning rate actually applies.
- Promotion gates compare candidate vs. best **with the same frozen value on both
  sides** — a pure policy comparison.

The first policy improved this way beats its own seed network decisively head-to-head
and gains ~+100 Elo against external engine ladders (Pikafish fixed-depth anchors).

## Chimera diagnostics

`tools/chimera_h2h_arena.py` plays head-spliced players against each other
(e.g. *A's policy + B's value* vs. *B*) to causally isolate **which head** carries a
regression or an improvement — surfacing changes that aggregate metrics (validation
loss, package-level gates) hide. `tools/aprime_mcts_vs_rawpolicy.py` measures whether
a network's own search beats its raw policy (whether self-play targets are worth
learning from). A paper about these findings is in preparation.

## Repository layout

```text
xiangqi_transformer_model.py   Transformer policy/value/WDL model
xiangqi_train.py               training loop, replay-buffer self-play ingest, anchors
xiangqi_selfplay.py            multi-process MCTS self-play (chimera-capable)
xiangqi_closed_loop.py         self-play → train → gate closed loop (v16 flywheel)
xiangqi_arena.py               promotion-gate arena (chimera-capable)
xiangqi_mcts_ext.py            Python wrapper for the C++ extension
xqcpp_ext_hist8_115.cpp        C++ board, movegen, tensor encoder, batched MCTS
xiangqi_model_battle_gui.py    visual GUI (Transformer vs. CNN baseline)
tools/                         arenas, ladders, diagnostics, experiment scripts
```

## Notes

- The C++ MCTS extension compiles automatically on first import via
  `torch.utils.cpp_extension` (needs a C++17 toolchain).
- Model checkpoints and training data are not tracked in git; see tagged releases
  (e.g. `V12`) for packaged checkpoints.
- External evaluation uses [Pikafish](https://github.com/official-pikafish/Pikafish)
  as a fixed-depth ladder; no absolute Elo is claimed.

## License

MIT — see [LICENSE](LICENSE).
