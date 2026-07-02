"""Board invariants: FEN round-trips, push/pop symmetry, and the fast
in-check routine cross-checked against the generator-based slow path."""

import random

from xiangqi_mcts_ext import Board


def test_fen_round_trip() -> None:
    board = Board()
    board.reset()
    fen = board.fen()
    other = Board()
    other.set_fen(fen)
    assert other.fen() == fen
    assert other.key() == board.key()


def test_push_pop_restores_state() -> None:
    board = Board()
    board.reset()
    rng = random.Random(20260703)
    for _ in range(40):
        fen = board.fen()
        key = board.key()
        moves = board.legal_moves()
        if not moves or board.is_game_over():
            break
        board.push(rng.choice(list(moves)))
        board.pop()
        assert board.fen() == fen
        assert board.key() == key
        # then actually advance so we test many positions, not just startpos
        board.push(rng.choice(list(moves)))


def test_fast_in_check_matches_slow_reference() -> None:
    """The optimized in_check must agree with the generator-based slow
    implementation across seeded random playouts (same cross-check the
    engine can enable at runtime via XQCPP_VERIFY_CHECK)."""
    rng = random.Random(115)
    for _game in range(5):
        board = Board()
        board.reset()
        for _ply in range(80):
            assert board.in_check_turn() == board.in_check_slow_turn()
            moves = board.legal_moves()
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(list(moves)))


def test_startpos_not_terminal() -> None:
    board = Board()
    board.reset()
    assert not board.is_game_over()
    assert not board.in_check_turn()
    assert board.plies_played() == 0
