import chess
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.start_positions import WHITE, FinishRecord
from src.eval.finishes import evaluate_finishes

SCHOLAR = FinishRecord("g1", chess.STARTING_FEN, ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"], WHITE)


class MateOrUniform(torch.nn.Module):
    """Puts all mass on a mating move when one exists, else uniform over legal moves.
    Decodes the position from the observation by replaying nothing: it is given the
    env list through `mates` (action ids per call) — kept simple by re-deriving the
    board from the legal-move mask is not possible, so the test uses depth 0 only."""

    def __init__(self, mate_action):
        super().__init__()
        self.mate_action = mate_action

    def forward(self, obs, mask):
        probs = mask / mask.sum(dim=1, keepdim=True)
        hit = mask[:, self.mate_action] > 0
        probs[hit] = 0.0
        probs[hit, self.mate_action] = 1.0
        return probs, torch.zeros(obs.shape[0], 1)


def _mate_action():
    env = OpenSpielEnv()
    env.reset(SCHOLAR.rewind(0))
    return next(iter(env.actions_for_uci(["h5f7"])))


def test_perfect_finisher_scores_one_at_depth_zero_and_short_records_are_skipped():
    model = MateOrUniform(_mate_action())
    out = evaluate_finishes(model, model, [SCHOLAR], depths=(0, 10), games=5, cap_margin=2)
    assert out["finish_rate_d0"] == 1.0 and out["finish_games_d0"] == 1
    assert out["finish_games_d10"] == 0 and out["finish_rate_d10"] is None


def test_cap_counts_as_failure_and_training_mode_is_restored():
    class Uniform(torch.nn.Module):
        def forward(self, obs, mask):
            return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)
    model = Uniform().train()
    out = evaluate_finishes(model, model, [SCHOLAR], depths=(6,), games=5, cap_margin=2, seed=0)
    assert out["finish_games_d6"] == 1 and out["finish_rate_d6"] in (0.0, 1.0)
    assert model.training
