# AlphaXiang — Chinese chess (Xiangqi) AlphaZero-style engine

**A deliverable-first Xiangqi (Chinese chess) RL project:** C++ MCTS core + PyTorch policy/value network, Apple Silicon friendly (**MPS + FP16**), reproducible runs, and a PVP client (clocks + undo).

---

## Highlights

- **Single trunk, dual-head network** (policy + value) in the AlphaGo/AlphaZero style.
- **Apple Silicon acceleration:** supports `MPS`.
- **Auto-resume checkpoints:** keeps writing and resumes from it on restart.
- **PVP client** with **chess clock + undo** for quick sanity checks and demos.

---

## Quick demo (PVP)

> Works best on macOS (M series chip). Run from PyCharm with **Ctrl+R** or from terminal.

Hotkeys:

- **Undo:**  `U` (undo 2 plies)
- **Reload latest weights:** `R`
- **Adjust strength:** `1 / 2 / 3` (MCTS sims)
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
- (Optional) `pygame` for the PVP UI

Python deps (example):

```
pip install -r requirements.txt
```
---

## Training (self-play)

Start training (**auto-resume** from pth file if present):

```bash
python AlphaXiang_v0.1.py
```

Key runtime notes:

- On Apple Silicon, `MPS` can speed up training.
- Self-play with `MPS` inference is possible but may be unstable with multi-process. The default setup prioritizes stability.

### Checkpoints

- Latest weights: `xiangqi_zero.pth`

If training “looks like it restarted,” verify:

- `xiangqi_zero.pth` exists and is being updated
- the logs show it loaded/resumed from `xiangqi_zero.pth`

---

## PVP (play vs the engine)

Our PVP script supports one-click defaults (engine path + latest weights), just run:

```bash
python xiangqi_pvp.py
```

---

## Performance & delivery notes

**Goal:** reduce iteration time while keeping correctness and reproducibility.

- Bottleneck targeted: search/inference hot path
- Reliability: auto-resume checkpoints + structured logs + reproducible build/run steps
- Verification: PVP client and debug-friendly telemetry (draw reasons, terminal types, etc.)

---

## Troubleshooting

### Too many draws (repetition / max steps)

This project includes history features and draw-handling logic. If draws dominate:

- increase repetition/no-progress penalties (if enabled)
- tune max steps or resign thresholds
- verify self-play randomness (seed diversity) is enabled

---

## Roadmap

- [ ] Use C++ to reconstruct the core code.
- [ ] Add arena to find the best model.
- [ ] Document training config presets (fast dev vs stronger play)
- [ ] Optional: add a web demo

---

## License

MIT (or your preferred license)

---

<details>
<summary><strong>Version française (FR)</strong> — résumé pour le contexte Canada</summary>

# AlphaXiang — Xiangqi (échecs chinois) style AlphaZero

Projet Xiangqi « deliverable-first » : cœur **MCTS (extension PyTorch), réseau **policy/value** en PyTorch, accélération **Apple Silicon (MPS)**, exécutions reproductibles, et client PVP (horloge + annuler).

## Points forts

- **Réseau à tronc partagé + deux têtes** (policy + value)
- **Reprise automatique** via `xiangqi_zero.pth`
- **Client PVP** avec **horloge + undo**

## Démarrage rapide

Entraînement :

```bash
python AlphaXiang_v0.1.py
```

Jeu PVP :

```bash
python xiangqi_pvp.py
```

</details>
