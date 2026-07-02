"""Perft: exhaustive move-generation counts against published reference values.

Perft is the standard way to prove a chess-variant move generator correct:
walk the full legal-move tree to a fixed depth and compare node counts with
independently published numbers. A single missing or extra move anywhere
(flying general, horse-leg blocks, river-crossing pawns, check evasions...)
changes the count.

Reference values for the Xiangqi starting position are the community-standard
perft numbers (see e.g. the Xiangqi perft results collected by the
computer-xiangqi community).
"""

import pytest

from xiangqi_mcts_ext import Board

# depth -> expected node count from the starting position
STARTPOS_PERFT = {
    1: 44,
    2: 1_920,
    3: 79_666,
}

STARTPOS_PERFT_SLOW = {
    4: 3_290_240,
}


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves():
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


@pytest.mark.parametrize("depth,expected", sorted(STARTPOS_PERFT.items()))
def test_perft_startpos(depth: int, expected: int) -> None:
    board = Board()
    board.reset()
    assert perft(board, depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize("depth,expected", sorted(STARTPOS_PERFT_SLOW.items()))
def test_perft_startpos_deep(depth: int, expected: int) -> None:
    board = Board()
    board.reset()
    assert perft(board, depth) == expected


def test_perft_is_side_effect_free() -> None:
    """A full perft walk must leave the board exactly as it found it."""
    board = Board()
    board.reset()
    fen_before = board.fen()
    key_before = board.key()
    perft(board, 2)
    assert board.fen() == fen_before
    assert board.key() == key_before
