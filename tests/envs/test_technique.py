import chess

from src.envs.start_positions import WHITE, BLACK
from src.envs.technique import TechniqueSampler, generate_position, _has_mate_in_one
import random


def test_queen_start_scenario():
    s = TechniqueSampler(["Q"], {"Q": 40}, seed=1)
    for _ in range(20):
        start = s.sample()
        b = chess.Board(start.fen)
        pieces = sorted(p.symbol().upper() for p in b.piece_map().values())
        assert pieces == ["K", "K", "Q"]
        strong = chess.WHITE if start.winner == WHITE else chess.BLACK
        assert b.turn == strong and b.piece_at(b.king(strong)).color == strong
        assert any(p.piece_type == chess.QUEEN and p.color == strong for p in b.piece_map().values())
        assert not _has_mate_in_one(b) and not b.is_check() and b.is_valid()
        assert start.cap == 40 and start.config == "Q" and start.depth == 0 and start.line == ()


def test_reproducible_and_uniform_over_sets():
    a = TechniqueSampler(["Q", "R", "RR", "QR"], seed=3)
    b = TechniqueSampler(["Q", "R", "RR", "QR"], seed=3)
    xs = [a.sample() for _ in range(30)]
    ys = [b.sample() for _ in range(30)]
    assert xs == ys
    assert {x.config for x in xs} == {"Q", "R", "RR", "QR"}
    assert all(x.cap == {"Q": 40, "R": 60, "RR": 40, "QR": 40}[x.config] for x in xs)


def test_strong_king_never_in_check_and_weak_king_not_attacked():
    rng = random.Random(0)
    for _ in range(50):
        b = generate_position("RR", rng)
        assert not b.is_check()
        assert not b.is_attacked_by(b.turn, b.king(not b.turn))


def test_bad_sets_rejected():
    import pytest
    with pytest.raises(ValueError):
        TechniqueSampler([], seed=0)
    with pytest.raises(ValueError):
        TechniqueSampler(["X"], seed=0)
