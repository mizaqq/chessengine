import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training


def test_eval_fn_called_at_interval_and_end():
    calls = []

    def eval_fn(white, black):
        calls.append(len(calls) + 1)
        return {"puzzle_top1": 0.0, "puzzle_mate_prob": 0.0}

    w, b = ChessPolicyProbs(num_filters=8), ChessPolicyProbs(num_filters=8)
    logs, *_ = run_chess_training(
        OpenSpielVectorEnv(1), w, b,
        torch.optim.Adam(w.parameters()), torch.optim.Adam(b.parameters()),
        episodes=12, steps=1, eval_fn=eval_fn, eval_interval=5, log_interval=10,
    )
    assert calls == [1, 2, 3]  # updates 5, 10, 12
    with_eval = [entry["episode"] for entry in logs if "puzzle_top1" in entry]
    assert with_eval == [5, 10, 12]


def test_no_eval_fn_logs_unchanged():
    w, b = ChessPolicyProbs(num_filters=8), ChessPolicyProbs(num_filters=8)
    logs, *_ = run_chess_training(
        OpenSpielVectorEnv(1), w, b,
        torch.optim.Adam(w.parameters()), torch.optim.Adam(b.parameters()),
        episodes=10, steps=1,
    )
    assert [e["episode"] for e in logs] == [10]
    assert "puzzle_top1" not in logs[0]
