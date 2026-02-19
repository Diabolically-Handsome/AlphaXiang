# AlphaXiang V11 - Xiangqi AlphaZero-Style Engine

Current V11 is a C++-accelerated Xiangqi training stack with self-play, arena gating, and a PVP GUI.

## Highlights

- C++ core extension (`xqcpp`) for board rules, legal moves, tensor export, and MCTS search
- PyTorch network with shared ResNet trunk + policy/value/material heads
- Sparse policy targets (`idxs + probs`) for efficient training
- Self-play workers with online weight reload
- Auto-resume from `latest_ckpt.pt` (model + optimizer + scheduler + step)
- Arena best-model gating (`current` vs `best`) before promotion
- LR scheduler + runtime LR recovery when LR falls to floor and loss stalls
- PVP GUI with clocks, byoyomi, undo, side swap, debug top-K, and hot-reload weights

## Repository Layout

- `Chessv11_cpp_hist8_115_mps_fp16.py` - trainer + self-play + arena loop
- `xqcpp_ext_hist8_115.cpp` - C++ extension source (xqcpp)
- `build_xqcpp115_v11.py` - optional manual prebuild for xqcpp
- `xiangqi_pvp_v11_2.py` - PVP / human-vs-AI GUI
- `v11_runs/` - checkpoints and snapshots

## Requirements

- Python 3.10+ (3.11 recommended)
- PyTorch
- NumPy
- `pygame` (for GUI)
- C++17 compiler toolchain for PyTorch C++ extension build
  - macOS: Xcode Command Line Tools
  - Linux: `build-essential` + Python dev headers
  - Windows: Visual Studio Build Tools (C++ workload)

## Quick Start

### 1) Train (self-play + learning)

```bash
python Chessv11_cpp_hist8_115_mps_fp16.py
```

Default behavior:

- Uses `CFG` inside the script (default model `128` channels / `20` residual blocks)
- Auto-resumes from `v11_runs/latest_ckpt.pt` if available
- Keeps `latest.pth`, `best.pth`, and step snapshots in `v11_runs/`

### 2) Optional: prebuild C++ extension

```bash
python build_xqcpp115_v11.py
```

### 3) Play PVP / Human-vs-AI

```bash
python xiangqi_pvp_v11_2.py --engine Chessv11_cpp_hist8_115_mps_fp16 --weights v11_runs/latest.pth --human red
```

Common options:

- `--human red|black|both`
- `--ai_device cpu|mps|cuda`
- `--sim_level 1|2|3`
- `--time_manager`

## PVP Hotkeys

- `U` / `Backspace`: undo
- `N`: new game
- `R`: reload weights
- `S`: swap side (human-vs-AI mode)
- `1` / `2` / `3`: AI strength
- `E`: exploration toggle
- `D`: debug overlay
- `ESC`: quit

## Current Training Defaults (V11)

- Input: `115` channels, `8` history frames
- Model: `128` channels, `20` residual blocks
- Replay buffer capacity: `270000`
- Batch size: `256`
- MCTS sims (self-play): `800`
- Arena: every `10000` train steps, `50` games, threshold `55%`
- Draw handling: weighted draw samples by default (`TRAIN_DRAW_MODE="weighted"`)

## Device Notes

- Trainer default: `TRAIN_DEVICE="mps"` with AMP enabled
- Self-play default: `SELFPLAY_DEVICE="mps"` + FP16
- CUDA is supported in device selection paths (`cpu/mps/cuda`) if your PyTorch build supports CUDA

Recommended:

- Multi-worker self-play: prefer CPU or reduce worker count when using MPS/CUDA
- If LR repeatedly decays to floor, use the built-in LR runtime recovery settings in `CFG`

## Output Files

Under `v11_runs/`:

- `latest.pth` - latest model weights
- `latest_ckpt.pt` - full resume checkpoint
- `best.pth` - current best model accepted by arena
- `model_stepXXXXXXX.pth` - archived snapshots
- `best_stepXXXXXXX.pth` - arena-accepted best snapshots

