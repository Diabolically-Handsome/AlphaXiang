# AlphaXiang v12.6-micro

AlphaXiang v12.6-micro is a Transformer-based AlphaZero-style Xiangqi engine.
This release bundles the strongest verified v12 checkpoint, the Transformer
model/training code, the PVP/battle GUI, and the C++ MCTS extension source.

## What Is Included

```text
models/
  alphaxiang_v12_6_micro_step296000.pt   strongest v12.6-micro checkpoint
  SHA256SUMS.txt                         checkpoint checksum

src/
  xiangqi_transformer_model.py           Transformer policy/value/WDL model
  xiangqi_train.py                       training loop and replay ingest
  xiangqi_model_battle_gui.py            visual GUI for Transformer-vs-CNN play
  xiangqi_mcts_ext.py                    Python wrapper for the C++ MCTS extension
  xqcpp_ext_hist8_115.cpp                C++ board, movegen, tensor encoder, MCTS
  CNN/                                   bundled CNN baseline used by the GUI
  tools/                                 optional model/opponent/arena utilities
```

## Checkpoint

- File: `models/alphaxiang_v12_6_micro_step296000.pt`
- Training step: `296000`
- Parameters: `38,610,182`
- Architecture: 12-layer Transformer, `d_model=512`, 8 heads, FFN 2048
- Input: 115 planes over a 10x9 Xiangqi board
- Heads: sparse policy over 8100 from-to actions, scalar value, WDL
- SHA256:
  `878d0c1c3a0cfa7ace4c5350c7b377450be3ff9838a1548bb0b129d89be82b9c`

## Verified Strength Snapshot

All results below use the bundled checkpoint with `sims=1600`, `q_weight=1.0`,
fixed openings, and side balancing where applicable.

| Opponent | Games | W-L-D | Score |
|---|---:|---:|---:|
| Pikafish d1 + 15% noise | 50 | 49-1-0 | 98.0% |
| Pikafish d3 | 50 | 18-21-11 | 47.0% |
| Pikafish d4 | 50 | 15-26-9 | 39.0% |
| Pikafish d5 | 50 | 3-37-10 | 16.0% |
| Fairy-Stockfish d3 | 50 | 46-2-2 | 94.0% |
| CNN best.pth | 50 | 49-0-1 | 99.0% |

These are short 50-game anchors, not a formal rating list.

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pygame
- A C++17 compiler compatible with PyTorch C++ extensions
- CUDA is recommended for practical search speed

The C++ extension is built by PyTorch the first time search is used. If that
build fails, install a matching compiler toolchain and verify that your PyTorch
installation can compile C++ extensions.

## Quick Start

From the release root:

```bash
python src/tools/model_opponent.py \
  --checkpoint models/alphaxiang_v12_6_micro_step296000.pt \
  --device cuda:0 \
  --sims 32
```

That command loads the checkpoint and runs a tiny start-position search smoke.

## Run The GUI

The bundled GUI defaults to a Transformer-vs-CNN visual match. From the release
root:

```bash
python src/xiangqi_model_battle_gui.py \
  --transformer-checkpoint models/alphaxiang_v12_6_micro_step296000.pt \
  --cnn-engine src/CNN/Chessv11_cpp_hist8_115_mps_fp16.py \
  --cnn-weights src/CNN/best.pth \
  --device cuda:0 \
  --num-simulations 800
```

Useful controls:

- `N`: new game
- `S`: swap sides
- `Space`: pause or resume
- `D`: debug overlay
- `R`: reload
- `ESC`: quit

## Training

The training entry point is:

```bash
python src/xiangqi_train.py --help
```

Training data is not included in this lightweight release. To resume or
finetune, provide your own `human_data_dir`, `selfplay_dirs`, and output
directory.

## Notes

- This package is intentionally focused on the strongest v12 release artifact.
- The source files are compatible with the bundled v12 checkpoint even though
  the codebase also contains forward-compatible options used by later v13 work.
- The checkpoint is large, so GitHub may require uploading it as a release
  asset rather than committing it directly to the repository.
