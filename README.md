# AlphaXiang — Chinese chess (Xiangqi) AlphaZero-style engine

**A deliverable-first Xiangqi (Chinese chess) RL project:** C++ MCTS core + PyTorch policy/value network, Apple Silicon friendly (**MPS + FP16**), reproducible runs, and a PVP client (clocks + undo).

> **TPM mindset:** measurable speedups, stable training loops, clear build steps, and “it runs on my machine” reliability.

---

## Highlights

- **C++-accelerated MCTS** via a PyTorch C++ extension (`xqcpp`) for performance-critical search.
- **Single trunk, dual-head network** (policy + value) in the AlphaGo/AlphaZero style.
- **8-frame history input (115 planes)** to better model repetition/no-progress dynamics.
- **Apple Silicon acceleration:** training supports `MPS` + mixed precision (**FP16 autocast**).
- **Auto-resume checkpoints:** keeps writing `v10_runs/latest.pth` and resumes from it on restart.
- **PVP client** with **chess clock + undo** for quick sanity checks and demos.

---

## Quick demo (PVP)

> Works best on macOS (M1/M2/M3). Run from PyCharm with **Ctrl+R** or from terminal.

Hotkeys:

- **Undo:** `Z` (undo 1 ply), `X` (undo 2 plies)
- **Reload latest weights:** `U`
- **Adjust strength:** `+ / -` (MCTS sims)
- **Restart:** `R` (play Red), `B` (play Black), `S` (swap sides)

---

## Repository layout

- `Chessv10_cpp_hist8_115_mps_fp16.py` — training + engine entry (PyTorch + self-play)
- `xqcpp_ext_hist8_115.cpp` — C++ extension (Board + MCTS search)
- `build_xqcpp115.py` — optional build helper
- `xiangqi_pvp.py` / `xiangqi_pvp_*` — PVP client(s)
- `v10_runs/` — checkpoints (`latest.pth`, `model_step*.pth`)

---

## Requirements

- Python 3.11+
- PyTorch with MPS support (macOS)
- **Xcode Command Line Tools** (needed to compile the C++ extension)
- (Optional) `pygame` for the PVP UI

Install Xcode CLT (macOS):

```bash
xcode-select --install
```

Python deps (example):

```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, a minimal set is typically: `torch`, `numpy`, `pygame` (and anything else you use).

---

## Build the C++ extension (xqcpp)

Most scripts will auto-build on first run. If you want to build explicitly:

```bash
python build_xqcpp115.py
```

If successful, you should see `xqcpp` built and loadable.

---

## Training (self-play)

Start training (**auto-resume** from `v10_runs/latest.pth` if present):

```bash
python Chessv10_cpp_hist8_115_mps_fp16.py
```

Key runtime notes:

- On Apple Silicon, `MPS + FP16` can speed up training.
- Self-play with `MPS` inference is possible but may be unstable with multi-process. The default setup prioritizes stability.

### Checkpoints

- Latest weights: `v10_runs/latest.pth`
- Step snapshots: `v10_runs/model_step000XXXX.pth`

If training “looks like it restarted,” verify:

- `latest.pth` exists and is being updated
- the logs show it loaded/resumed from `latest.pth`

---

## PVP (play vs the engine)

If your PVP script supports one-click defaults (engine path + latest weights), just run:

```bash
python xiangqi_pvp.py
```

Or specify weights explicitly if your script accepts it:

```bash
python xiangqi_pvp.py --weights v10_runs/latest.pth
```

---

## Performance & delivery notes (TPM style)

**Goal:** reduce iteration time while keeping correctness and reproducibility.

- Bottleneck targeted: search/inference hot path
- Approach: move MCTS core into a C++ extension + optimize evaluation batching
- Reliability: auto-resume checkpoints + structured logs + reproducible build/run steps
- Verification: PVP client and debug-friendly telemetry (draw reasons, terminal types, etc.)

> If you want to add a numbers section, include 1–2 concrete metrics like “self-play time per N games” before/after.

---

## Troubleshooting

### “xqcpp extension not found / cannot compile”

- Make sure Xcode CLT is installed:

```bash
xcode-select --install
```

- Check that `xqcpp_ext_hist8_115.cpp` is in the project root.
- Rebuild:

```bash
python build_xqcpp115.py
```

### “MPS op not supported” / Mixed precision issues

Try disabling FP16/mixed precision (set config flag if supported), or enable fallback:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Too many draws (repetition / max steps)

This project includes history features and draw-handling logic. If draws dominate:

- increase repetition/no-progress penalties (if enabled)
- tune max steps or resign thresholds
- verify self-play randomness (seed diversity) is enabled

---

## Roadmap

- [ ] Add CI (build + lint + minimal smoke test)
- [ ] Add small evaluation harness (fixed-seed test suite)
- [ ] Document training config presets (fast dev vs stronger play)
- [ ] Optional: add a lightweight web demo

---

## License

MIT (or your preferred license)

---

<details>
<summary><strong>Version française (FR)</strong> — résumé pour le contexte Canada</summary>

# AlphaXiang — Xiangqi (échecs chinois) style AlphaZero

Projet Xiangqi « deliverable-first » : cœur **MCTS en C++** (extension PyTorch), réseau **policy/value** en PyTorch, accélération **Apple Silicon (MPS + FP16)**, exécutions reproductibles, et client PVP (horloge + annuler).

## Points forts

- **MCTS accéléré en C++** via l’extension `xqcpp`
- **Réseau à tronc partagé + deux têtes** (policy + value)
- **Historique 8 frames (115 plans)** pour mieux gérer répétitions / no-progress
- **Reprise automatique** via `v10_runs/latest.pth`
- **Client PVP** avec **horloge + undo**

## Démarrage rapide

Installer **Xcode Command Line Tools** :

```bash
xcode-select --install
```

Entraînement :

```bash
python Chessv10_cpp_hist8_115_mps_fp16.py
```

Jeu PVP :

```bash
python xiangqi_pvp.py
```

## Dépannage (court)

Si `xqcpp` ne compile pas : vérifier Xcode CLT, puis

```bash
python build_xqcpp115.py
```

</details>
