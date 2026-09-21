import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import StartPositionSampler, Puzzle
from src.model.chess_model import ChessPolicyProbs
from src.model.training import run_chess_training, PPO_PRESET
from src.training.metrics import MetricsAggregator


def _run(scores, patience=2):
    torch.manual_seed(0)
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(2, puzzles, 2)
    model = ChessPolicyProbs(num_filters=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    seen = {}
    it = iter(scores)
    def eval_fn(w, b):
        s = next(it)
        seen[len(seen)] = {k: v.clone() for k, v in w.state_dict().items()}
        return {"prior_score": s}
    logs, _, model, _ = run_chess_training(env, model, None, opt, None, metrics=MetricsAggregator(), episodes=6, steps=4,
                                           terminal_rewards={"win": 2, "loss": -2, "draw": -0.5}, gamma=0.99, log_interval=1,
                                           eval_fn=eval_fn, eval_interval=1, algorithm=PPO_PRESET,
                                           stop_below_prior=0.5, stop_patience=patience)
    return logs, model, seen


def test_alarm_stops_and_restores_last_good_weights():
    logs, model, seen = _run([0.6, 0.35, 0.38, 0.7, 0.7, 0.7])
    assert logs[-1]["episode"] == 3 and logs[-1]["alarm_stop"] == 3
    good = seen[0]                      # weights at the evaluation that scored 0.6 (update 1)
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            assert torch.equal(v, good[k])


def test_single_dip_does_not_stop():
    logs, _, _ = _run([0.6, 0.35, 0.55, 0.6, 0.6, 0.6])
    assert logs[-1]["episode"] == 6 and "alarm_stop" not in logs[-1]
