import chess
import pytest

from src.envs.start_positions import WHITE, FinishCurriculum, FinishRecord

SCHOLAR = FinishRecord("g1", chess.STARTING_FEN,
                       ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"], WHITE)


def test_rewind_zero_is_the_position_before_the_mate_with_winner_to_move():
    b = chess.Board(SCHOLAR.rewind(0))
    assert b.turn == chess.WHITE
    b.push_uci("h5f7")
    assert b.is_checkmate()


def test_rewind_even_depth_keeps_winner_to_move():
    assert chess.Board(SCHOLAR.rewind(2)).turn == chess.WHITE
    assert SCHOLAR.rewind(6) == chess.STARTING_FEN


def test_sample_depths_are_even_and_clipped_to_the_record():
    cur = FinishCurriculum([SCHOLAR], depth_start=10, depth_max=40, cap_margin=20, seed=1)
    starts = [cur.sample() for _ in range(50)]
    assert all(s.depth % 2 == 0 and s.depth <= 6 for s in starts)
    assert all(s.cap == s.depth + 20 and s.winner == WHITE for s in starts)
    assert {s.depth for s in starts} == {0, 2, 4, 6}


def test_advance_at_threshold_and_not_below():
    cur = FinishCurriculum([SCHOLAR], depth_start=2, window=100, advance_rate=0.6)
    for i in range(100):
        cur.report(2, i < 59)
    assert cur.current_depth == 2
    cur = FinishCurriculum([SCHOLAR], depth_start=2, window=100, advance_rate=0.6)
    for i in range(100):
        cur.report(2, i < 61)
    assert cur.current_depth == 4
    starts = {cur.sample().depth for _ in range(100)}
    assert starts == {0, 2, 4}


def test_needs_a_full_window_since_the_last_rise_and_respects_ceiling():
    cur = FinishCurriculum([SCHOLAR], depth_start=2, depth_max=4, window=10, advance_rate=0.5)
    for _ in range(10):
        cur.report(2, True)
    assert cur.current_depth == 4
    for _ in range(9):
        cur.report(4, True)
    assert cur.current_depth == 4          # window not yet refilled, and at ceiling anyway
    for _ in range(20):
        cur.report(4, True)
    assert cur.current_depth == 4


def test_validation():
    with pytest.raises(ValueError):
        FinishCurriculum([], 2)
    with pytest.raises(ValueError):
        FinishCurriculum([SCHOLAR], depth_start=3)
    with pytest.raises(ValueError):
        FinishCurriculum([SCHOLAR], advance_rate=0.0)
    with pytest.raises(ValueError):
        FinishCurriculum([SCHOLAR], cap_margin=1)
