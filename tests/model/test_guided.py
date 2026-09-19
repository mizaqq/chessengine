import math

import chess
import torch

from src.envs.open_spiel_env import OpenSpielEnv
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.envs.start_positions import WHITE, FinishCurriculum, FinishRecord, Puzzle, StartPositionSampler
from src.model.training import _collect_rollout, guided_probs
from src.training.metrics import MetricsAggregator

SCHOLAR = FinishRecord("g1", chess.STARTING_FEN, ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"], WHITE)


def test_guided_probs_mixture_and_rows_without_demo():
    p = torch.tensor([[0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
    b = guided_probs(p, [{2}, set()], 0.25)
    assert torch.allclose(b[0], torch.tensor([0.675, 0.0375, 0.2875]))
    assert torch.equal(b[1], p[1])
    assert abs(b[0].sum().item() - 1.0) < 1e-6


def test_puzzle_board_demo_is_the_mating_move_and_finishing_board_follows_the_human_line():
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    cur = FinishCurriculum([SCHOLAR], depth_start=4, depth_max=4)
    env = OpenSpielVectorEnv(3, puzzles, 1, None, None, 0, cur, 1)
    env.reset()
    demos = env.demo_actions()
    assert demos[0] == env.envs[0].actions_for_uci(["a1a8"])          # puzzle: rules mate
    line = env.finish_start[1].line                                   # human moves from the rewound position
    assert demos[1] == env.envs[1].actions_for_uci([line[0]])         # finishing: human's next move
    assert demos[2] == set()                                          # opening board: nothing
    if len(line) >= 3:
        # follow the line one ply, then deviate with a quiet move: the demo falls back to rules
        env.envs[1].step(next(iter(demos[1])))
        assert env.envs[1].demo_actions() == env.envs[1].actions_for_uci([line[1]])
        quiet = next(iter(env.envs[1].actions_for_uci(["a7a6", "h7h6"])))
        env.envs[1].step(quiet)
        assert not env.envs[1].demo_alive
        assert env.envs[1].demo_actions() == set(env.envs[1].actions_for_uci([])) | set(
            __import__("src.envs.open_spiel_env", fromlist=["mating_actions_of"]).mating_actions_of(env.envs[1]))


def test_rollout_stores_mixture_log_prob_and_counts_demo_picks():
    class Uniform(torch.nn.Module):
        def forward(self, obs, mask):
            return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    torch.manual_seed(0)
    env = OpenSpielVectorEnv(2, puzzles, 1)
    step = env.reset()
    m = MetricsAggregator()
    guide = {"epsilon": 0.5, "mask": torch.tensor([True, False])}
    steps, _ = _collect_rollout(env, Uniform(), Uniform(), step, 1, 2, {"win": 2, "loss": -2, "draw": -0.5}, m,
                                oriented=True, guide=guide)
    n_legal = int(step.legal_actions_mask[0].sum())
    a = int(steps[0].action[0]); mate = next(iter(env.envs[0].actions_for_uci(["a1a8"]))) if False else None
    lp = steps[0].old_log_prob[0].item()
    # the played move has probability 0.5/n_legal (+0.5 if it is the mate)
    assert abs(lp - math.log(0.5 / n_legal)) < 1e-4 or abs(lp - math.log(0.5 / n_legal + 0.5)) < 1e-4
    assert steps[0].noisy.tolist() == [True, False]
    s = m.episode_summary()
    assert s["guide_moves"] == 1 and s["guide_demo_share"] in (0.0, 1.0)


def test_guided_sampling_flags_only_demonstrator_fires_and_keeps_mixture_log_prob():
    """With epsilon 0 the demonstrator never fires (all episodes clean) and the stored
    log-prob is the network's; with epsilon ~1 it always fires and the move is a demo move."""
    from src.envs.start_positions import FinishStart, WHITE

    class Uniform(torch.nn.Module):
        def forward(self, obs, mask):
            return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)

    class Rec:
        current_depth = 0
        def __init__(self, s): self.s, self.calls = s, []
        def sample(self): return self.s
        def report(self, d, s): pass
        def report_label(self, label, success, clean=True, depth=None): self.calls.append(clean)
        def levels(self): return {"Q": 0}

    fen = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"      # Qg7 is mate in one
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    for eps, expect_clean in ((0.0, True), (0.999, False)):
        rec = Rec(FinishStart(fen, 0, 4, WHITE, (), "Q"))
        env = OpenSpielVectorEnv(1, puzzles, 0, None, None, 0, rec, 1)
        step = env.reset()
        torch.manual_seed(3)
        m = MetricsAggregator()
        guide = {"epsilon": eps, "mask": torch.tensor([True])} if eps > 0 else None
        steps, _ = _collect_rollout(env, Uniform(), Uniform(), step, 4, 1, {"win": 2, "loss": -2, "draw": -0.5}, m,
                                    oriented=True, guide=guide)
        if eps > 0:
            mate = next(iter(env.envs[0].actions_for_uci(["g1g7"]))) if False else None
            n_legal = int(step.legal_actions_mask[0].sum())
            # the first ply is the demo move with mixture prob ~ eps + (1-eps)/n_legal
            assert abs(steps[0].old_log_prob[0].item() - math.log(eps + (1 - eps) / n_legal)) < 1e-3
            assert m.episode_summary()["technique_clean_attempts_Q"] == 0
        assert rec.calls and all(c is expect_clean for c in rec.calls[:1])


class _Uniform(torch.nn.Module):
    def forward(self, obs, mask):
        return mask / mask.sum(dim=1, keepdim=True), torch.zeros(obs.shape[0], 1)


class _Rec:
    current_depth = 0
    def __init__(self, s): self.s = s
    def sample(self): return self.s
    def report(self, d, s): pass
    def report_label(self, label, success, clean=True, depth=None): pass
    def levels(self): return {"Q": 0}


def _one_board(fen, winner, line=()):
    from src.envs.start_positions import FinishStart
    puzzles = StartPositionSampler([Puzzle("p", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], 0)])
    env = OpenSpielVectorEnv(1, puzzles, 0, None, None, 0, _Rec(FinishStart(fen, 0, 40, winner, line, "Q")), 1)
    return env, env.reset()


def test_punisher_takes_hung_queen_with_mixture_log_prob_and_flags_episode():
    """Black to move, white queen on d4 hangs to the c6 knight; punish_epsilon 0.5 gives
    Nxd4 behaviour probability 0.5/n + 0.5, and the stored log-prob is the mixture's."""
    from src.envs.start_positions import BLACK
    fen = "4k3/p7/2n5/8/3Q4/8/8/4K3 b - - 0 1"
    for eps in (0.5, 0.999):
        env, step = _one_board(fen, BLACK)
        nxd4 = next(iter(env.envs[0].actions_for_uci(["c6d4"])))
        n_legal = int(step.legal_actions_mask[0].sum())
        torch.manual_seed(1)
        m = MetricsAggregator()
        punish = {"epsilon": eps, "threshold": 3, "mask": torch.tensor([True])}
        steps, _ = _collect_rollout(env, _Uniform(), _Uniform(), step, 1, 1, {"win": 2, "loss": -2, "draw": -0.5}, m,
                                    oriented=True, guide=None, punish=punish)
        a, lp = int(steps[0].action[0]), steps[0].old_log_prob[0].item()
        expected = math.log((1 - eps) / n_legal + (eps if a == nxd4 else 0.0))
        assert abs(lp - expected) < 1e-4
        s = m.episode_summary()
        assert s["punish_moves"] == 1 and s["guide_moves"] == 0
        if eps > 0.99:
            assert a == nxd4 and s["punish_picks"] == 1 and env._demo_acted[0] is True


def test_label_demonstration_wins_over_punisher():
    """A finishing board still on the human line keeps guide_epsilon even when a free
    capture exists."""
    from src.envs.start_positions import BLACK
    fen = "4k3/p7/2n5/8/3Q4/8/8/4K3 b - - 0 1"
    env, step = _one_board(fen, BLACK, line=("e8d8",))       # the "human" plays Kd8?? — not a capture
    kd8 = env.envs[0].actions_for_uci(["e8d8"])
    if not kd8:                                                 # Kd8 illegal here (queen covers d8): use Kf8
        env, step = _one_board(fen, BLACK, line=("e8f8",))
        kd8 = env.envs[0].actions_for_uci(["e8f8"])
    demo = next(iter(kd8))
    n_legal = int(step.legal_actions_mask[0].sum())
    torch.manual_seed(2)
    m = MetricsAggregator()
    guide = {"epsilon": 0.25, "mask": torch.tensor([True])}
    punish = {"epsilon": 0.999, "threshold": 3, "mask": torch.tensor([True])}
    steps, _ = _collect_rollout(env, _Uniform(), _Uniform(), step, 1, 1, {"win": 2, "loss": -2, "draw": -0.5}, m,
                                oriented=True, guide=guide, punish=punish)
    a, lp = int(steps[0].action[0]), steps[0].old_log_prob[0].item()
    assert abs(lp - math.log(0.75 / n_legal + (0.25 if a == demo else 0.0))) < 1e-4
    s = m.episode_summary()
    assert s["guide_moves"] == 1 and s["punish_moves"] == 0


def test_punisher_off_reproduces_guided_rollout_exactly():
    from src.envs.start_positions import WHITE
    fen = "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"
    outs = []
    for punish in (None, {"epsilon": 0.0, "threshold": 3, "mask": torch.tensor([True])}):
        env, step = _one_board(fen, WHITE)
        torch.manual_seed(5)
        steps, _ = _collect_rollout(env, _Uniform(), _Uniform(), step, 3, 1, {"win": 2, "loss": -2, "draw": -0.5},
                                    MetricsAggregator(), oriented=True,
                                    guide={"epsilon": 0.25, "mask": torch.tensor([True])}, punish=punish)
        outs.append([(int(s.action[0]), round(s.old_log_prob[0].item(), 6)) for s in steps])
    assert outs[0] == outs[1]
