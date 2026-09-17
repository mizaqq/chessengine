import pytest
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


def test_entropy_metrics_accumulate_and_average():
    m = MetricsAggregator()
    m.add_entropy(raw=2.0, normalized=0.8)
    m.add_entropy(raw=1.0, normalized=0.6)
    result = m.episode_summary()
    assert result["mean_entropy_raw"] == 1.5
    assert result["mean_entropy_normalized"] == 0.7


def test_entropy_metrics_reset_between_windows():
    m = MetricsAggregator()
    m.add_entropy(raw=2.0, normalized=0.8)
    first = m.episode_summary()
    assert first["mean_entropy_raw"] == 2.0

    m.add_entropy(raw=1.0, normalized=0.5)
    second = m.episode_summary()
    assert second["mean_entropy_raw"] == 1.0
    assert second["mean_entropy_normalized"] == 0.5


def test_entropy_metrics_zero_when_no_data():
    m = MetricsAggregator()
    result = m.episode_summary()
    assert result["mean_entropy_raw"] == 0.0
    assert result["mean_entropy_normalized"] == 0.0


def test_mean_terminal_return_averages_finished_games():
    m = MetricsAggregator()
    m.add_terminal_return(2.0)
    m.add_terminal_return(-2.0)
    m.add_terminal_return(-0.5)
    result = m.episode_summary()
    assert result["mean_terminal_return"] == pytest.approx(-0.5 / 3)


def test_mean_terminal_return_is_none_without_finished_games():
    m = MetricsAggregator()
    result = m.episode_summary()
    assert result["mean_terminal_return"] is None


def test_mean_terminal_return_resets_between_windows():
    m = MetricsAggregator()
    m.add_terminal_return(2.0)
    m.episode_summary()
    assert m.episode_summary()["mean_terminal_return"] is None


def test_puzzle_results_split_by_depth():
    m = MetricsAggregator()
    m.add_puzzle_result(True, depth=1); m.add_puzzle_result(False, depth=1)
    m.add_puzzle_result(False, depth=2); m.add_puzzle_result(False, depth=2); m.add_puzzle_result(True, depth=2); m.add_puzzle_result(False, depth=2)
    s = m.episode_summary()
    assert s["puzzle_solved_rate"] == 2 / 6
    assert s["puzzle_solved_rate_m1"] == 0.5 and s["puzzle_solved_rate_m2"] == 0.25
    assert s["puzzle_attempts_m2"] == 4
    assert "puzzle_solved_rate_m1" not in m.episode_summary()   # window reset


def test_finish_results_split_by_depth_and_stay_out_of_game_rates():
    m = MetricsAggregator()
    m.set_finish_depth(2)
    m.add_finish_result(0, True)
    m.add_finish_result(2, True)
    m.add_finish_result(2, False)
    m.add_terminal_result(draw=True)
    s = m.episode_summary()
    assert s["finish_attempts"] == 3
    assert abs(s["finish_success_rate"] - 2 / 3) < 1e-9
    assert s["finish_success_rate_d0"] == 1.0 and s["finish_success_rate_d2"] == 0.5
    assert s["finish_attempts_d2"] == 2 and s["finish_depth"] == 2
    assert s["draw_rate"] == 1.0
    s2 = m.episode_summary()
    assert s2["finish_attempts"] == 0 and s2["finish_success_rate"] is None and s2["finish_depth"] == 2


def test_explore_share():
    m = MetricsAggregator()
    m.add_explore(count=10, offprior=3)
    s = m.episode_summary()
    assert s["explore_moves"] == 10 and abs(s["explore_offprior_share"] - 0.3) < 1e-9
    assert m.episode_summary()["explore_offprior_share"] is None
