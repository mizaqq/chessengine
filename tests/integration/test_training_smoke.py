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


PUZZLE_ROWS = """puzzle_id,fen,mating_moves,rating
p1,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,a1a8,600
p2,7k/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,a1a8,600
p3,r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1,a8a1,600
"""


def test_training_smoke_with_puzzle_curriculum_and_eval(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text(PUZZLE_ROWS)
    config = {
        "num_envs": 2,
        "max_updates": 2,
        "steps_per_update": 2,
        "seed": 1,
        "env_type": "sync",
        "puzzle_fraction": 1.0,
        "puzzle_train_file": str(path),
        "puzzle_eval_file": str(path),
        "eval_interval": 1,
    }
    result = run_training_from_config(config)
    logs = result["logs"]
    assert [e["episode"] for e in logs] == [1, 2]
    for entry in logs:
        assert entry["puzzle_count"] == 3
        assert 0.0 <= entry["puzzle_top1"] <= 1.0
        assert 0.0 <= entry["puzzle_mate_prob"] <= 1.0
