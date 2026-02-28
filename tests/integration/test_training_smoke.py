import torch
from src.entrypoints.train import run_training_from_config


def test_training_smoke_runs_short_job_without_nan():
    """
    Smoke test: run short training updates and verify no NaN metrics.
    This protects against orchestration regressions.
    """
    config = {
        "num_envs": 2,
        "max_updates": 12,
        "steps_per_update": 3,
        "seed": 42,
        "log_interval": 10,
    }
    
    result = run_training_from_config(config)
    
    assert result is not None
    assert "logs" in result
    assert "losses" in result
    
    assert len(result["losses"]) > 0, "No losses recorded"
    
    for loss in result["losses"]:
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
    
    if len(result["logs"]) > 0:
        for log_entry in result["logs"]:
            if "loss" in log_entry:
                assert not torch.isnan(torch.tensor(log_entry["loss"])), "Loss is NaN"
            if "entropy" in log_entry:
                assert not torch.isnan(torch.tensor(log_entry["entropy"])), "Entropy is NaN"
