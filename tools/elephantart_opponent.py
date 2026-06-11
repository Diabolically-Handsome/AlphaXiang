"""ElephantArt UCCI opponent wrapper.

ElephantArt is an open-source AlphaZero-style xiangqi engine (CNN + MCTS,
trained via self-play, no human knowledge — fundamentally different paradigm
from Pikafish/Fairy-Stockfish which use NNUE + alpha-beta).  This makes it a
strong held-out evaluator: if our Pikafish-trained model still beats it, our
generalisation is real; if it tanks vs ElephantArt while beating Pikafish,
we know the model is overfitting to NNUE-style play.

Setup notes (one-time):
    1. Clone source: git clone https://github.com/CGLemon/ElephantArt
    2. Build: requires cmake (`pip install cmake` works) and g++.  See README.
       (Patch needed: add `#include <cstdint>` to src/FileSystem.h on modern GCC.)
    3. Download weights from the README's Google Drive folder
       (`gdown --folder ...` works for public folders).
    4. Default binary path used here: /home/laure/engines/ElephantArt/build/Elephant
    5. Default weights:                /home/laure/engines/ElephantArt/weights/NN/trained-5-23-45000games-4b-256c.txt

CLI flags (set automatically):
    -m ucci          → UCCI protocol mode (matches Pikafish notation)
    -w <weights>     → weights file
    -t <threads>     → search threads
    -p <playouts>    → MCTS playouts per move (~strength knob)
    -b <batch>       → NN batch size

ElephantArt does NOT understand `setoption name Threads/Hash value ...` —
all config is via CLI flags.  We tell PikafishOpponent to skip the Threads/Hash
setoption lines via `send_threads_and_hash=False`.

Usage:
    eng = make_elephantart_opponent(playouts=800)
    eng.new_game()
    eng.set_position(START_FEN)
    best, _ = eng.go_depth(4)  # depth is ignored — search depth is bounded by --playouts
"""
from __future__ import annotations

from pathlib import Path

from pikafish_opponent import PikafishOpponent


_DEFAULT_BINARY_PATH = "/home/laure/engines/ElephantArt/build/Elephant"
_DEFAULT_WEIGHTS_PATH = (
    "/home/laure/engines/ElephantArt/weights/NN/trained-5-23-45000games-4b-256c.txt"
)


def make_elephantart_opponent(
    binary_path: str | Path = _DEFAULT_BINARY_PATH,
    weights_path: str | Path = _DEFAULT_WEIGHTS_PATH,
    threads: int = 2,
    playouts: int = 800,
    batch_size: int = 1,
    startup_timeout_s: float = 15.0,
) -> PikafishOpponent:
    """Construct a PikafishOpponent-style wrapper around ElephantArt.

    Returns an object with the same interface as PikafishOpponent
    (new_game / set_position / go_depth / go_movetime / go_nodes / close).
    Note that go_depth's depth argument is effectively ignored — ElephantArt
    sizes its search by `--playouts`.  For per-game strength control, set
    `playouts` at construction time."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"ElephantArt weights not found: {weights_path}")

    extra_argv = (
        "-m", "ucci",
        "-w", str(weights_path),
        "-t", str(int(threads)),
        "-p", str(int(playouts)),
        "-b", str(int(batch_size)),
    )
    return PikafishOpponent(
        binary_path=binary_path,
        threads=threads,
        hash_mb=0,  # unused (ElephantArt has no Hash setoption)
        startup_timeout_s=startup_timeout_s,
        extra_argv=extra_argv,
        handshake="ucci",
        send_threads_and_hash=False,
    )


def _self_test() -> None:
    """Smoke test: handshake + one search from start position."""
    START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    print("starting ElephantArt (UCCI, playouts=400)...")
    eng = make_elephantart_opponent(playouts=400)
    try:
        eng.new_game()
        eng.set_position(START_FEN)

        best, ponder = eng.go_depth(4)  # depth is moot, --playouts caps the search
        print(f"playouts=400 from startpos: bestmove={best} ponder={ponder}")
        assert len(best) == 4, f"expected 4-char UCI move, got {best!r}"

        # Black response sanity
        eng.set_position(START_FEN, moves=["h2e2"])
        best2, _ = eng.go_depth(4)
        print(f"after red h2e2: black bestmove={best2}")
        assert best2[1] in "56789", (
            f"black move should originate in rank 5-9, got {best2!r}"
        )
        print("PASS — ElephantArt integration works")
    finally:
        eng.close()


if __name__ == "__main__":
    _self_test()
