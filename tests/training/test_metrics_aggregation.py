from src.training.metrics import MetricsAggregator


def test_episode_counters_accumulate_over_multiple_steps():
    m = MetricsAggregator()
    m.add_step(empty_masks=2, illegal_samples=1)
    m.add_step(empty_masks=1, illegal_samples=3)
    result = m.episode_summary()
    assert result["episode_empty_masks"] == 3
    assert result["episode_illegal_samples"] == 4
