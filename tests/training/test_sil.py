import torch

from src.core.types import StepRecord
from src.training.sil import EpisodeAccumulator, ReplayBuffer, sil_loss, WHITE, BLACK


def _step(players, actions, values, done, tr_w, tr_b, learner=None):
    n = len(players)
    obs = torch.zeros(n, 20, 8, 8); obs[:, 0, 0, 0] = 1
    legal = torch.zeros(n, 4674); legal[:, :4] = 1
    return StepRecord(player=torch.tensor(players), obs=obs, legal_mask=legal, action=torch.tensor(actions),
                      old_log_prob=torch.zeros(n), old_value=torch.tensor(values), entropy=torch.zeros(n),
                      num_legal=torch.full((n,), 4.0), potential_white=torch.zeros(n), done=torch.tensor(done),
                      terminal_r_white=torch.tensor(tr_w), terminal_r_black=torch.tensor(tr_b),
                      learner=None if learner is None else torch.tensor(learner))


def test_three_ply_win_returns_scenario():
    buf = ReplayBuffer(10)
    acc = EpisodeAccumulator(1, 0.99, buf)
    acc.add_step(_step([WHITE], [0], [0.0], [False], [0.0], [0.0]))
    acc.add_step(_step([BLACK], [1], [0.0], [False], [0.0], [0.0]))
    pushed = acc.add_step(_step([WHITE], [2], [0.0], [True], [2.0], [-2.0]))
    assert pushed == 3 and len(buf) == 3
    assert [round(float(r), 4) for r in buf.ret[:3]] == [round(2 * 0.99 ** 2, 4), round(-2 * 0.99, 4), round(2 * 0.99, 4)]
    assert list(buf.action[:3]) == [0, 1, 2]
    # priorities: (R - V)+ with V = 0 -> the losing black ply has priority 0
    assert buf.priority[1] == 0 and buf.priority[0] > 0 and buf.priority[2] > 0


def test_opponent_plies_excluded_and_ring_drops_oldest():
    buf = ReplayBuffer(2)
    acc = EpisodeAccumulator(1, 1.0, buf)
    acc.add_step(_step([WHITE], [0], [0.0], [False], [0.0], [0.0], learner=[False]))   # opponent ply
    acc.add_step(_step([BLACK], [1], [0.0], [False], [0.0], [0.0], learner=[True]))
    acc.add_step(_step([WHITE], [2], [0.0], [False], [0.0], [0.0], learner=[False]))
    acc.add_step(_step([BLACK], [3], [0.0], [True], [-2.0], [2.0], learner=[True]))
    assert len(buf) == 2 and sorted(buf.action[:2].tolist()) == [1, 3]
    acc.add_step(_step([WHITE], [7], [0.0], [True], [2.0], [-2.0], learner=[True]))
    assert len(buf) == 2 and 7 in buf.action[:2].tolist() and 1 not in buf.action[:2].tolist()


def test_only_better_than_expected_plies_train():
    class V(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(4674))
            self.v = torch.nn.Parameter(torch.tensor([0.5, 0.0]))

        def forward(self, obs, mask):
            p = torch.softmax(self.logits, 0) * mask
            p = p / p.sum(dim=1, keepdim=True)
            return p.expand(obs.shape[0], -1), self.v[: obs.shape[0]].unsqueeze(-1)

    m = V()
    obs = torch.zeros(2, 20, 8, 8); mask = torch.zeros(2, 4674); mask[:, :4] = 1
    loss, adv, stats = sil_loss(m, obs, mask, torch.tensor([0, 1]), torch.tensor([1.5, -1.0]), torch.ones(2), 0.01)
    assert adv.tolist() == [1.0, 0.0] and stats["sil_valid_share"] == 0.5
    loss.backward()
    # the value head gets gradient only at the first ply
    assert m.v.grad[0] != 0 and m.v.grad[1] == 0
    # policy term equals -log pi(a0) * 1.0 / 2 (mean over two plies, second is zero)
    expected = -torch.log(torch.tensor(0.25)) * 1.0 / 2
    assert abs(stats["sil_policy_loss"] - expected.item()) < 1e-5


def test_sampling_skips_zero_priority_and_returns_none_when_all_zero():
    buf = ReplayBuffer(4, seed=0)
    obs = torch.zeros(20, 8, 8); mask = torch.zeros(4674); mask[:4] = 1
    buf.add(obs, mask, 0, ret=-1.0, value=0.0)
    assert buf.sample(3) is None
    buf.add(obs, mask, 1, ret=1.0, value=0.0)
    idx, w = buf.sample(3)
    assert set(idx.tolist()) == {1} and w.shape[0] == len(idx)
    o, mk, a, r = buf.get(idx)
    assert o.shape[1:] == (20, 8, 8) and mk.shape[1] == 4674 and a.tolist() == [1] * len(idx)


def test_positive_share_counts_whole_buffer():
    buf = ReplayBuffer(4, seed=0)
    obs = torch.zeros(20, 8, 8); mask = torch.zeros(4674); mask[:4] = 1
    buf.add(obs, mask, 0, ret=-1.0, value=0.0)
    buf.add(obs, mask, 1, ret=1.0, value=0.0)
    assert buf.positive_share() == 0.5
