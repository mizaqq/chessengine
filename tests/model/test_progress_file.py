import json
import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import StartPositionSampler, Puzzle
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training, PPO_PRESET
from src.training.metrics import MetricsAggregator


def test_progress_file_written_each_log_interval(tmp_path):
    torch.manual_seed(0)
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(2, puzzles, 2)
    model = ChessPolicyProbs(num_filters=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "progress.json"
    run_chess_training(env, model, None, opt, None, metrics=MetricsAggregator(), episodes=3, steps=4,
                       terminal_rewards={"win": 2, "loss": -2, "draw": -0.5}, gamma=0.99, log_interval=1,
                       algorithm=PPO_PRESET, progress_path=str(path))
    d = json.loads(path.read_text())
    assert d["episode"] == 3 and d["episodes"] == 3 and len(d["logs"]) == 3 and "updated" in d
