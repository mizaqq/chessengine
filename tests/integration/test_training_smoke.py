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
        "puzzle_fraction": 0.5,
        "puzzle_train_file": str(path),
        "puzzle_eval_file": str(path),
        "eval_interval": 1,
        "log_interval": 1,
    }
    result = run_training_from_config(config)
    logs = result["logs"]
    assert [e["episode"] for e in logs] == [1, 2]
    assert logs[-1]["puzzle_attempts"] >= 1  # one puzzle board, one attempt per ply
    for entry in logs:
        assert entry["puzzle_count"] == 3
        assert 0.0 <= entry["puzzle_top1"] <= 1.0
        assert 0.0 <= entry["puzzle_mate_prob"] <= 1.0


def test_training_smoke_ppo_logs_diagnostics():
    config = {
        "num_envs": 2, "max_updates": 3, "steps_per_update": 3, "seed": 3, "env_type": "sync",
        "algorithm": "ppo", "ppo_epochs": 2, "ppo_minibatches": 2, "log_interval": 1,
    }
    result = run_training_from_config(config)
    for loss in result["losses"]:
        assert not torch.isnan(torch.tensor(loss))
    for entry in result["logs"]:
        assert isinstance(entry["mean_clip_fraction"], float)
        assert isinstance(entry["mean_approx_kl"], float)
        assert isinstance(entry["mean_value_loss"], float)


def test_training_smoke_a2c_has_null_clip_fraction():
    result = run_training_from_config({"num_envs": 2, "max_updates": 1, "steps_per_update": 2, "log_interval": 1})
    assert result["logs"][-1]["mean_clip_fraction"] is None
    assert isinstance(result["logs"][-1]["mean_approx_kl"], float)


def test_continuation_restores_optimizer_state_when_present(tmp_path, capsys):
    from src.model.chess_model import ChessPolicyProbs
    from src.model.checkpoints import save_model
    m = ChessPolicyProbs(num_filters=8, num_blocks=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    (sum(p.sum() for p in m.parameters())).backward(); opt.step()          # one real Adam step
    save_model(m, tmp_path, updates=1)
    torch.save(opt.state_dict(), tmp_path / "optimizer.pth")
    config = {"num_envs": 2, "max_updates": 1, "steps_per_update": 2, "seed": 3, "env_type": "sync",
              "init_from": str(tmp_path), "device": "cpu",
              "terminal_rewards": {"win": 2.0, "loss": -2.0, "draw": -0.5}}
    result = run_training_from_config(config)
    assert "optimizer state restored" in capsys.readouterr().out
    steps = [st["step"] for st in result["optimizer"].state_dict()["state"].values()]
    assert steps and min(float(s) for s in steps) >= 2          # the saved step (1) plus this run's
    config["resume_optimizer"] = False
    result2 = run_training_from_config(config)
    steps2 = [float(st["step"]) for st in result2["optimizer"].state_dict()["state"].values()]
    assert max(steps2) == 1
