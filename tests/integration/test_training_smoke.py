import torch
from src.entrypoints.train import run_training_from_config


def test_training_smoke_runs_short_job_without_nan():
    config = {
        "num_envs": 2,
        "max_updates": 12,
        "steps_per_update": 3,
        "seed": 42,
        "env_type": "sync",
        "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5},
    }
    result = run_training_from_config(config)

    assert result is not None
    assert "logs" in result
    assert "losses" in result
    assert len(result["losses"]) > 0, "No losses recorded"

    for loss in result["losses"]:
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
