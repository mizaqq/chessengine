import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import Puzzle
from src.eval.puzzles import evaluate_mate_in_one, mating_actions

ROOK_MATE_W = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
ROOK_MATE_B = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"
PUZZLES = [Puzzle("w", ROOK_MATE_W, ["a1a8"], 600), Puzzle("b", ROOK_MATE_B, ["a8a1"], 600)]


class UniformPolicy(torch.nn.Module):
    def forward(self, obs, legal):
        probs = legal / legal.sum(dim=1, keepdim=True)
        return probs, torch.zeros(obs.shape[0], 1)


class PerfectPolicy(torch.nn.Module):
    """Puts all mass on the (single) mating move of each rook-mate position."""
    def forward(self, obs, legal):
        probs = torch.zeros_like(legal)
        for i in range(obs.shape[0]):
            # Both rook-mate positions have 20 legal moves; white's mate is action 22.
            env = OpenSpielEnv()
            env.reset(ROOK_MATE_W if legal[i, 22] == 1 else ROOK_MATE_B)
            (mate,) = mating_actions(env)
            probs[i, mate] = 1.0
        return probs, torch.zeros(obs.shape[0], 1)


def test_mating_actions_single_rook_mate():
    env = OpenSpielEnv()
    env.reset(ROOK_MATE_W)
    assert len(mating_actions(env)) == 1


def test_perfect_policy_scores_one():
    out = evaluate_mate_in_one(PerfectPolicy(), PerfectPolicy(), PUZZLES)
    assert out["puzzle_top1"] == 1.0
    assert abs(out["puzzle_mate_prob"] - 1.0) < 1e-6
    assert out["puzzle_count"] == 2


def test_uniform_policy_on_twenty_legal_moves():
    out = evaluate_mate_in_one(UniformPolicy(), UniformPolicy(), PUZZLES[:1])
    assert abs(out["puzzle_mate_prob"] - 0.05) < 1e-6
    assert out["puzzle_top1"] in (0.0, 1.0)


def test_evaluation_restores_training_mode_and_rng():
    w, b = UniformPolicy(), UniformPolicy()
    w.train(); b.eval()
    before = torch.rand(1).item()
    torch.manual_seed(5)
    evaluate_mate_in_one(w, b, PUZZLES)
    after_eval = torch.rand(1).item()
    torch.manual_seed(5)
    assert torch.rand(1).item() == after_eval
    assert w.training and not b.training
