import torch

from src.model.training import _precompute_terminal_rewards, WHITE, BLACK
from src.training.metrics import MetricsAggregator

TR = {"win": 2.0, "loss": -2.0, "draw": -0.5, "puzzle_miss": -0.3}


def test_puzzle_miss_pays_mover_only():
    players = torch.tensor([WHITE, BLACK, WHITE])
    info = {"game_results": {0: "puzzle_miss", 1: "puzzle_miss", 2: "white_win"}}
    tr_w, tr_b = _precompute_terminal_rewards(info, 3, TR, players)
    assert torch.allclose(tr_w, torch.tensor([-0.3, 0.0, 2.0]))
    assert torch.allclose(tr_b, torch.tensor([0.0, -0.3, -2.0]))


def test_puzzle_miss_defaults_to_zero():
    players = torch.tensor([WHITE])
    tr_w, tr_b = _precompute_terminal_rewards(
        {"game_results": {0: "puzzle_miss"}}, 1, {"win": 2.0, "loss": -2.0, "draw": -0.5}, players
    )
    assert tr_w.tolist() == [0.0] and tr_b.tolist() == [0.0]


def test_metrics_split_puzzle_and_opening_games():
    m = MetricsAggregator()
    for i in range(30):
        m.add_puzzle_result(solved=i < 3)
    m.add_terminal_result(draw=True)
    m.add_terminal_result(draw=True)
    s = m.episode_summary()
    assert s["puzzle_attempts"] == 30
    assert abs(s["puzzle_solved_rate"] - 0.1) < 1e-9
    assert s["total_games"] == 2
    assert s["draw_rate"] == 1.0
    assert m.episode_summary()["puzzle_attempts"] == 0  # window reset
