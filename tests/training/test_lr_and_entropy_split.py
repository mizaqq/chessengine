import pytest
import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import Puzzle, StartPositionSampler
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training
from src.training.metrics import MetricsAggregator

ROOK_MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


def _run(envs, **kw):
    w, b = ChessPolicyProbs(num_filters=8), ChessPolicyProbs(num_filters=8)
    ow, ob = torch.optim.Adam(w.parameters(), lr=1e-3), torch.optim.Adam(b.parameters(), lr=1e-3)
    logs, *_ = run_chess_training(envs, w, b, ow, ob, steps=1, **kw)
    return logs, ow


def test_lr_decays_by_factor_to_floor():
    _, opt = _run(OpenSpielVectorEnv(1), episodes=4, lr_decay_interval=1, lr_decay_factor=0.5, min_lr=2e-4)
    # 1e-3 -> 5e-4 -> 2.5e-4 -> floor 2e-4 -> 2e-4
    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-12


def test_min_lr_above_start_is_rejected():
    from src.entrypoints.train import run_training_from_config
    with pytest.raises(ValueError, match="min_lr"):
        run_training_from_config({"num_envs": 1, "max_updates": 1, "steps_per_update": 1,
                                  "learning_rate": 1e-4, "min_lr": 3e-4})


def test_entropy_split_by_board_type():
    sampler = StartPositionSampler([Puzzle("p", ROOK_MATE, ["a1a8"], 600)], seed=0)
    envs = OpenSpielVectorEnv(2, sampler, num_puzzle_envs=1)
    logs, _ = _run(envs, episodes=10, log_interval=10)
    entry = logs[-1]
    assert entry["mean_entropy_normalized_game"] is not None
    assert entry["mean_entropy_normalized_puzzle"] is not None
    assert 0.0 <= entry["mean_entropy_normalized_game"] <= 1.0


def test_entropy_split_absent_without_puzzle_boards():
    logs, _ = _run(OpenSpielVectorEnv(1), episodes=10, log_interval=10)
    assert logs[-1]["mean_entropy_normalized_puzzle"] is None
    assert logs[-1]["mean_entropy_normalized_game"] is not None


def test_metrics_aggregator_split_means():
    m = MetricsAggregator()
    m.add_entropy(1.0, 0.5, game=0.8, puzzle=0.2)
    m.add_entropy(1.0, 0.5, game=0.6, puzzle=None)
    s = m.episode_summary()
    assert abs(s["mean_entropy_normalized_game"] - 0.7) < 1e-9
    assert abs(s["mean_entropy_normalized_puzzle"] - 0.2) < 1e-9
