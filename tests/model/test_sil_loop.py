"""Self-imitation in the training loop: off leaves the update sequence identical; on
produces the diagnostics and a non-empty buffer."""
import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import StartPositionSampler, Puzzle
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training, PPO_PRESET
from src.training.metrics import MetricsAggregator


def _run(sil, seed=0):
    torch.manual_seed(seed)
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(2, puzzles, 2)
    model = ChessPolicyProbs(num_filters=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    m = MetricsAggregator()
    logs, _, model, _ = run_chess_training(env, model, None, opt, None, metrics=m, episodes=3, steps=4,
                                           algorithm={**PPO_PRESET, "epochs": 1, "minibatches": 1},
                                           log_interval=1, sil=sil)
    return logs, model


def test_sil_off_matches_plain_loop_and_on_reports_diagnostics():
    logs_a, model_a = _run(None, seed=1)
    logs_b, model_b = _run({"updates": 0}, seed=1)
    assert all(torch.equal(p, q) for p, q in zip(model_a.parameters(), model_b.parameters()))
    assert "sil_buffer_size" not in logs_a[-1]
    logs_c, model_c = _run({"updates": 2, "batch": 8, "loss_weight": 0.1, "value_weight": 0.01, "buffer": 100}, seed=1)
    assert logs_c[-1]["sil_buffer_size"] > 0
    assert "sil_valid_share" in logs_c[-1]
    assert not all(torch.equal(p, q) for p, q in zip(model_a.parameters(), model_c.parameters()))
