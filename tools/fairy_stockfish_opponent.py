"""Fairy-Stockfish (xiangqi mode) UCI/UCCI opponent wrapper.

Fairy-Stockfish is an alpha-beta + NNUE engine with built-in xiangqi support.
It is **architecturally distinct from Pikafish** (different NNUE network, different
training data, different search heuristics) — making it a useful held-out
evaluation opponent that exposes Pikafish-style overfitting in our model.

By default Fairy-SF uses 1-based bottom-up rank notation (red back rank = 1),
which doesn't match Pikafish's 0-based convention.  We switch its `Protocol`
option to `ucci`, which makes it use 0-based ranks identical to Pikafish.
After this switch, the same `PikafishOpponent` UCI subprocess machinery works
without any move-translation layer.

Setup notes (one-time):
    1. Download the prebuilt xiangqi binary:
       https://github.com/fairy-stockfish/Fairy-Stockfish/releases
       File: fairy-stockfish-largeboard_x86-64-bmi2 (release fairy_sf_14_0_1_xq)
    2. Place at /home/laure/engines/fairy-stockfish-xq (or override --binary)
    3. The xiangqi NNUE weights (xiangqi-83f16c17fe26.nnue) are bundled into
       the binary, no separate download needed.

Usage (mirrors PikafishOpponent):
    pf = make_fairy_stockfish_opponent()
    pf.new_game()
    pf.set_position(START_FEN)
    best, ponder = pf.go_depth(6)
"""
from __future__ import annotations

from pathlib import Path

from pikafish_opponent import PikafishOpponent


_DEFAULT_BINARY_PATH = "/home/laure/engines/fairy-stockfish-xq"

_XIANGQI_UCCI_OPTIONS = (
    # Engage xiangqi rules + NNUE
    "setoption name UCI_Variant value xiangqi",
    # Switch from default UCI rank notation (1-based) to UCCI (0-based, matches Pikafish)
    "setoption name Protocol value ucci",
)


def make_fairy_stockfish_opponent(
    binary_path: str | Path = _DEFAULT_BINARY_PATH,
    threads: int = 1,
    hash_mb: int = 64,
    startup_timeout_s: float = 10.0,
) -> PikafishOpponent:
    """Construct a PikafishOpponent-like wrapper around Fairy-Stockfish.

    Returns a `PikafishOpponent` instance configured for xiangqi+UCCI mode.
    The instance can be used identically to a real PikafishOpponent.
    """
    return PikafishOpponent(
        binary_path=binary_path,
        threads=threads,
        hash_mb=hash_mb,
        startup_timeout_s=startup_timeout_s,
        extra_setoption_lines=_XIANGQI_UCCI_OPTIONS,
    )


def _self_test() -> None:
    """Smoke test: handshake + one search from start position."""
    START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    print("starting Fairy-Stockfish (xiangqi+ucci)...")
    eng = make_fairy_stockfish_opponent()
    try:
        eng.new_game()
        eng.set_position(START_FEN)

        # Test depth-based search
        best, ponder = eng.go_depth(6)
        print(f"depth=6 from startpos: bestmove={best} ponder={ponder}")
        assert len(best) == 4, f"expected 4-char UCI move, got {best!r}"

        # Test that black response is sane after red's central cannon
        eng.set_position(START_FEN, moves=["h2e2"])
        best2, _ = eng.go_depth(6)
        print(f"depth=6 after red h2e2: black bestmove={best2}")
        # Black's typical response to central cannon is one of: h7e7, b9c7, h9g7, etc.
        # Just sanity-check it's a valid 4-char move starting in black's half (rank 5-9)
        assert best2[1] in "56789", (
            f"black move should originate in rank 5-9, got {best2!r}"
        )
        print("PASS — Fairy-Stockfish integration works")
    finally:
        eng.close()


if __name__ == "__main__":
    _self_test()
