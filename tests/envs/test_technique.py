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
        assert start.cap == 40 and start.config == "Q" and start.depth == 3 and start.line == ()


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


def test_level_zero_start_scenario():
    s = TechniqueSampler(["Q"], {"Q": 40}, seed=5, curriculum=True, level_start=0)
    for _ in range(15):
        start = s.sample()
        assert start.depth == 0 and start.config == "Q"
        b = chess.Board(start.fen)
        strong = b.turn
        weak_k, strong_k = b.king(not strong), b.king(strong)
        assert weak_k in (chess.A1, chess.A8, chess.H1, chess.H8)
        assert chess.square_distance(weak_k, strong_k) == 2
        queen = next(sq for sq, p in b.piece_map().items() if p.piece_type == chess.QUEEN)
        assert chess.square_distance(weak_k, queen) <= 4
        assert not _has_mate_in_one(b) and not b.is_check()


def test_curriculum_advances_and_window_resets():
    s = TechniqueSampler(["Q", "R"], seed=0, curriculum=True, level_start=0, advance_rate=0.6, window=30)
    for i in range(30):
        s.report_label("Q", i < 20)          # 20 of 30 won
    assert s.current_level("Q") == 1 and s.current_level("R") == 0
    assert len(s._recent["Q"]) == 0          # window restarted
    for i in range(29):
        s.report_label("Q", True)
    assert s.current_level("Q") == 1         # not yet a full window
    s.report_label("Q", True)
    assert s.current_level("Q") == 2
    for _ in range(200):
        s.report_label("Q", True)
    assert s.current_level("Q") == 3         # ceiling
    assert s.levels() == {"Q": 3, "R": 0}


def test_without_curriculum_every_start_is_level_three_and_reports_ignored():
    s = TechniqueSampler(["Q"], seed=0)
    assert all(s.sample().depth == 3 for _ in range(5))
    for _ in range(40):
        s.report_label("Q", True)
    assert s.current_level("Q") == 3


def test_curriculum_draws_levels_up_to_current_only():
    s = TechniqueSampler(["R"], seed=2, curriculum=True, level_start=1)
    levels = {s.sample().depth for _ in range(40)}
    assert levels <= {0, 1} and 0 in levels and 1 in levels


def test_teacher_made_wins_do_not_count():
    s = TechniqueSampler(["Q"], seed=0, curriculum=True, level_start=0, advance_rate=0.7, window=40)
    for _ in range(60):
        s.report_label("Q", True, clean=False)
    assert s.current_level("Q") == 0 and len(s._recent["Q"]) == 0
    for i in range(40):
        s.report_label("Q", i < 28, clean=True)        # 28 of 40 clean wins
    assert s.current_level("Q") == 1
