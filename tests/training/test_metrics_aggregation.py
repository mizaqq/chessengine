from src.training.metrics import MetricsAggregator


def test_step_counters_accumulate_within_window():
    m = MetricsAggregator()
    m.add_step(empty_masks=2, illegal_samples=1)
    m.add_step(empty_masks=1, illegal_samples=3)
    result = m.episode_summary()
    assert result["empty_masks"] == 3
    assert result["illegal_samples"] == 4


def test_terminal_results_produce_win_rates():
    m = MetricsAggregator()
    m.add_terminal_result(white_win=True)
    m.add_terminal_result(white_win=True)
    m.add_terminal_result(black_win=True)
    m.add_terminal_result(draw=True)
    result = m.episode_summary()
    assert result["total_games"] == 4
    assert result["white_wins"] == 2
    assert result["white_win_rate"] == 0.5
    assert result["black_win_rate"] == 0.25
    assert result["draw_rate"] == 0.25


def test_summary_resets_counters_for_next_window():
    m = MetricsAggregator()
    m.add_step(empty_masks=5, illegal_samples=2)
    m.add_terminal_result(white_win=True)
    first = m.episode_summary()
    assert first["white_wins"] == 1
    assert first["empty_masks"] == 5

    m.add_step(empty_masks=1, illegal_samples=0)
    m.add_terminal_result(black_win=True)
    second = m.episode_summary()
    assert second["white_wins"] == 0
    assert second["black_wins"] == 1
    assert second["empty_masks"] == 1


def test_summary_with_no_games_returns_zero_rates():
    m = MetricsAggregator()
    result = m.episode_summary()
    assert result["total_games"] == 0
    assert result["white_win_rate"] == 0.0
