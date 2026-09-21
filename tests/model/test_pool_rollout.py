"""Opponent pool in the rollout: routing, learner mask, gather filter, bootstrap sign."""
import torch

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import StartPositionSampler, Puzzle
from src.model.chess_model import ChessPolicyProbs
from src.model.training import _collect_rollout, _gather_side, _compute_bootstrap
from src.training.metrics import MetricsAggregator
from src.training.opponent_pool import OpponentPool, PoolBoards, WHITE, BLACK

TR = {"win": 2, "loss": -2, "draw": -0.5}


class Const(torch.nn.Module):
    """Uniform policy, constant value, so the two networks are distinguishable."""
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs, mask):
        return mask / mask.sum(dim=1, keepdim=True), torch.full((obs.shape[0], 1), self.value)


def make_env(n=2):
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    torch.manual_seed(0)
    env = OpenSpielVectorEnv(n, puzzles, 0)
    return env, env.reset()


def test_learner_is_black_on_pool_board_only_black_plies_reach_batch():
    env, step = make_env(2)
    learner, opponent = Const(0.25), Const(-0.75)
    pool = OpponentPool(opponent, size=0, snapshot_every=50, seed=0)
    boards = PoolBoards([1], pool, seed=0)
    boards._state[1] = ("prior", pool.members[0][1], BLACK)
    m = MetricsAggregator()
    steps, end = _collect_rollout(env, learner, learner, step, 4, 2, TR, m, oriented=True, pool=boards)
    # board 1 alternates white (opponent), black (learner), white, black
    assert [bool(s.learner[1]) for s in steps] == [False, True, False, True]
    assert all(bool(s.learner[0]) for s in steps)                      # self-play board untouched
    assert [round(s.old_value[1].item(), 2) for s in steps] == [-0.75, 0.25, -0.75, 0.25]
    adv = [torch.zeros(2) for _ in steps]
    g_w = _gather_side(steps, adv, adv, WHITE)
    g_b = _gather_side(steps, adv, adv, BLACK)
    assert g_w["obs"].shape[0] == 2      # board 0's two white plies only
    assert g_b["obs"].shape[0] == 4      # board 0's two + board 1's two black plies
    assert torch.all(g_b["old_value"] == 0.25) and torch.all(g_w["old_value"] == 0.25)
    # bootstrap for black with the opponent (white) to move on board 1: learner's own value, negated
    boot = _compute_bootstrap(end, learner, learner, model_id=BLACK, oriented=True)
    assert end.current_player[1] == WHITE and abs(boot[1].item() + 0.25) < 1e-6


def test_pool_off_matches_plain_rollout():
    env1, s1 = make_env(2)
    env2, s2 = make_env(2)
    net = Const(0.0)
    torch.manual_seed(1); a, _ = _collect_rollout(env1, net, net, s1, 3, 2, TR, MetricsAggregator(), oriented=True)
    torch.manual_seed(1); b, _ = _collect_rollout(env2, net, net, s2, 3, 2, TR, MetricsAggregator(), oriented=True, pool=None)
    assert all(torch.equal(x.action, y.action) for x, y in zip(a, b))
    assert all(bool(x.learner.all()) for x in a)


def test_pool_game_result_scored_from_learner_side_and_redrawn():
    m = MetricsAggregator()
    m.add_pool_result("prior", 1.0)
    m.add_pool_result("prior", 0.5)
    s = m.episode_summary()
    assert s["pool_games"] == 2 and s["pool_score"] == 0.75 and s["pool_score_prior"] == 0.75
    assert s["total_games"] == 0


def test_pool_referee_replaces_opponent_ply_with_the_punishing_move():
    from src.envs.start_positions import FenSampler
    fen = "4k3/p7/8/8/3q4/1N6/8/4K3 w - - 0 1"          # white (pool opponent) to move: Nxd4 wins the queen
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(2, puzzles, 0, game_start_sampler=FenSampler([fen]), num_midgame_envs=2)
    step = env.reset()
    nxd4 = next(iter(env.envs[1].actions_for_uci(["b3d4"])))
    learner, opponent = Const(0.25), Const(-0.75)
    pool = OpponentPool(opponent, size=0, snapshot_every=50, seed=0)
    boards = PoolBoards([1], pool, seed=0)
    boards._state[1] = ("prior", pool.members[0][1], BLACK)
    m = MetricsAggregator()
    torch.manual_seed(0)
    steps, _ = _collect_rollout(env, learner, learner, step, 1, 2, TR, m, oriented=True, pool=boards, referee={"threshold": 3})
    assert int(steps[0].action[1]) == nxd4                # referee took the queen on the pool board
    assert not bool(steps[0].learner[1])                  # and it was the opponent's ply, not trained
    s = m.episode_summary()
    assert s["referee_moves"] == 1 and s["referee_picks"] == 1 and s["punish_moves"] == 0
