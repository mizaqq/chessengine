"""Technique starts in the finishing slot: label in info, counted apart, cap = draw."""
import chess
import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import StartPositionSampler, Puzzle, FinishStart, WHITE
from src.envs.technique import TechniqueSampler
from src.eval.technique import evaluate_technique, technique_starts
from src.model.training import _collect_rollout
from src.training.metrics import MetricsAggregator


class Uniform(torch.nn.Module):
    def forward(self, obs, mask):
        return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)


class FixedStart:
    """Sampler that always returns one labelled start."""
    current_depth = 0

    def __init__(self, start):
        self.start = start

    def sample(self):
        return self.start

    def report(self, depth, success):
        pass


def test_labelled_start_counted_apart_and_cap_is_a_draw():
    # queen v king, white to move, cap 2 plies: a uniform policy will not mate; the cap ends it
    start = FinishStart("7k/8/8/8/8/8/1Q6/K7 w - - 0 1", 0, 2, WHITE, (), "Q")
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(1, puzzles, 0, None, None, 0, FixedStart(start), 1)
    step = env.reset()
    m = MetricsAggregator()
    torch.manual_seed(0)
    _collect_rollout(env, Uniform(), Uniform(), step, 3, 1, {"win": 2, "loss": -2, "draw": -0.5}, m, oriented=True)
    s = m.episode_summary()
    assert s["technique_attempts_Q"] >= 1 and s["technique_success_rate_Q"] == 0.0
    assert s["finish_attempts"] == 0 and s["total_games"] == 0


def test_env_info_carries_label():
    start = FinishStart("7k/8/8/8/8/8/1Q6/K7 w - - 0 1", 0, 1, WHITE, (), "Q")
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(1, puzzles, 0, None, None, 0, FixedStart(start), 1)
    env.reset()
    legal = env.envs[0].get_legal_actions().nonzero().squeeze(1)
    step = env.step(torch.tensor([int(legal[0])]))
    assert step.info["finish_config"] == {0: "Q"} and step.info["finish_boards"] == {0: True}


def test_evaluation_uses_fixed_positions_and_reports_keys():
    a = technique_starts("R", 5, 60, seed=7)
    b = technique_starts("R", 5, 60, seed=7)
    assert [x.fen for x in a] == [x.fen for x in b]
    out = evaluate_technique(Uniform(), Uniform(), ["Q"], {"Q": 4}, games=3, oriented=True, seed=0)
    assert set(out) == {"technique_rate_Q", "technique_games_Q"} and out["technique_games_Q"] == 3
