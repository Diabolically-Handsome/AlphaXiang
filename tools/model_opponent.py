"""ModelOpponent: a frozen-snapshot of our model used as a self-play opponent.

Goal: provide a callable that takes a Board and returns a move (int), so that
``pikafish_selfplay.py:play_one_game`` can plug it in for the opponent's turns
instead of a Pikafish UCI subprocess.

Why
---
v6/v8 demonstrated that pure-Pikafish curricula over-specialize.  v7's mixed
Pikafish curriculum helped a lot, but plateaued at weighted Elo ~1717.  Adding
self-play games (model vs frozen-self) should inject a fundamentally different
training distribution: the opponent has the same *style* as us, mistakes that
WE make, etc.  Different from any external engine.

Design
------
* Loads the snapshot model once, runs ``make_gpu_evaluator`` on it.
* Each ``search(board, seed)`` call runs ``mcts_search`` against the snapshot's
  evaluator and returns the best move.
* A single ModelOpponent instance is meant to be SHARED across all worker
  threads in pikafish_selfplay's ThreadPoolExecutor — the underlying evaluator
  serializes inference internally, similar to how cross_game_batcher works.
  We do NOT recreate per-thread because (a) it 16x's VRAM usage on the
  selfplay GPU and (b) PyTorch model.forward is thread-safe in eval mode.

Opponent search settings
------------------------
For self-play data quality, the opponent should be roughly *as strong* as us
or slightly weaker.  By default we use sims=400 (vs our typical 800) — this
gives the opponent meaningful tactical play but lets our side win sometimes
(positive training signal).  Configurable via the ``sims`` arg.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

# Allow standalone import (e.g. running this file's _self_test).  When imported
# from tools/pikafish_selfplay.py the path is already wired up; this is a no-op.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xiangqi_mcts_ext import make_gpu_evaluator, mcts_search  # noqa: E402
from xiangqi_transformer_model import build_model_from_checkpoint_state  # noqa: E402


class ModelOpponent:
    """Frozen-snapshot model wrapped as an MCTS opponent."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda:1",
        sims: int = 400,
        c_puct: float = 1.25,
        temperature_move: float = 0.1,  # near-argmax; opponent plays its best
        eval_batch_size: int = 16,
        use_bfloat16: bool = True,
        max_plies: int = 300,
        repeat_limit: int = 6,
        repeat_min_ply: int = 30,
        no_capture_limit: int = 60,
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"opponent checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise RuntimeError(
                f"unsupported opponent checkpoint format at {ckpt_path}; "
                "expected a dict with 'model_state_dict' key"
            )

        model = build_model_from_checkpoint_state(state)
        model.to(device).eval()

        self._device = torch.device(device)
        self.evaluator = make_gpu_evaluator(
            model=model, device=str(device), use_bfloat16=use_bfloat16,
        )
        # Hold a reference to the model so it doesn't get GC'd while evaluator
        # holds a (presumably) raw pointer to it.
        self._model = model
        self.checkpoint_step = int(state.get("global_step", 0))
        self.checkpoint_path = str(ckpt_path)

        # MCTS hyperparameters
        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.temperature_move = float(temperature_move)
        self.eval_batch_size = int(eval_batch_size)
        self.max_plies = int(max_plies)
        self.repeat_limit = int(repeat_limit)
        self.repeat_min_ply = int(repeat_min_ply)
        self.no_capture_limit = int(no_capture_limit)

    # ------------------------------------------------------------------
    # Game-time interface
    # ------------------------------------------------------------------

    def search(self, board, seed: int) -> int:
        """Run MCTS on the given Board and return the best move (int).

        ``board`` is a live xiangqi_mcts_ext.Board.  ``seed`` should be unique
        per game-and-ply so concurrent threads don't all pick the same move.
        """
        best_move, _idxs, _probs, _root_v = mcts_search(
            board=board,
            net=self.evaluator,
            num_simulations=self.sims,
            c_puct=self.c_puct,
            q_weight=1.0,
            q_clip=1.0,
            add_root_noise=False,           # opponent plays deterministically
            dirichlet_alpha=0.3,
            dirichlet_eps=0.0,
            temperature_move=self.temperature_move,
            temperature_target=1.0,
            eval_batch_size=self.eval_batch_size,
            seed=int(seed),
            canonical_input=True,
            canonical_policy=True,
            max_plies=self.max_plies,
            repeat_limit=self.repeat_limit,
            repeat_min_ply=self.repeat_min_ply,
            no_capture_limit=self.no_capture_limit,
        )
        return int(best_move)

    def close(self) -> None:
        """Release references so the GPU memory can be GC'd."""
        self.evaluator = None
        self._model = None

    def __enter__(self) -> "ModelOpponent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _self_test() -> None:
    """Smoke test: load a checkpoint, run one search from start position."""
    import argparse
    from xiangqi_mcts_ext import Board

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="snapshot to use as opponent")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--sims", type=int, default=400)
    args = p.parse_args()

    print(f"loading opponent: {args.checkpoint}")
    opp = ModelOpponent(args.checkpoint, device=args.device, sims=args.sims)
    print(f"  step={opp.checkpoint_step}")

    board = Board()
    move = opp.search(board, seed=123)
    print(f"opponent search from startpos (sims={args.sims}): bestmove={move}")
    assert move >= 0, f"expected legal move, got {move}"
    opp.close()
    print("PASS — ModelOpponent works")


if __name__ == "__main__":
    _self_test()
