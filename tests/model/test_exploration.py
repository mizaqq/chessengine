import math

import torch

from src.model.training import behaviour_probs


def test_epsilon_zero_returns_the_policy():
    p = torch.tensor([[0.5, 0.5, 0.0]])
    legal = torch.tensor([[1.0, 1.0, 0.0]])
    assert torch.equal(behaviour_probs(p, legal, 0.0, 0.3), p)


def test_noise_only_on_legal_moves_and_rows_sum_to_one():
    torch.manual_seed(0)
    p = torch.zeros(4, 4674)
    legal = torch.zeros(4, 4674)
    for i in range(4):
        legal[i, [3, 100, 4000]] = 1.0
        p[i, 3] = 1.0                        # confident on one move
    b = behaviour_probs(p, legal, 0.25, 0.3)
    assert torch.allclose(b.sum(dim=1), torch.ones(4), atol=1e-6)
    assert (b[:, legal[0] == 0] == 0).all()
    assert (b[:, 3] >= 0.75).all() and (b[:, 3] < 1.0).all()
    assert (b[:, [100, 4000]].sum(dim=1) > 0).all()


def test_spec_scenario_log_prob_of_mixture():
    # pi(a) = 0.01, noise(a) = 0.20, eps = 0.25  ->  b(a) = 0.0575
    p = torch.tensor([[0.01, 0.99]])
    noise = torch.tensor([[0.20, 0.80]])
    b = 0.75 * p + 0.25 * noise
    assert math.isclose(math.log(b[0, 0].item()), math.log(0.0575), rel_tol=1e-6)
    assert math.log(0.0575) > math.log(0.01)


def test_rollout_stores_behaviour_log_prob():
    """With eps 1 and a huge alpha the behaviour is ~uniform over legal moves, so the
    stored log-prob is ~log(1/num_legal) rather than log pi(a)."""
    from src.model.training import _collect_rollout
    from src.training.metrics import MetricsAggregator
    from src.envs.open_spiel_vector_env import OpenSpielVectorEnv

    class Peaky(torch.nn.Module):
        def forward(self, obs, mask):
            first = mask.argmax(dim=1)
            probs = torch.full_like(mask, 1e-6) * mask
            probs[torch.arange(len(first)), first] = 1.0
            probs = probs / probs.sum(dim=1, keepdim=True)
            return probs, torch.zeros(obs.shape[0], 1)

    torch.manual_seed(0)
    env = OpenSpielVectorEnv(2)
    step = env.reset()
    m = MetricsAggregator()
    explore = {"epsilon": 1.0, "alpha": 1e6, "mask": torch.tensor([True, False])}
    steps, _ = _collect_rollout(env, Peaky(), Peaky(), step, 1, 2, {"win": 2, "loss": -2, "draw": -0.5}, m,
                                oriented=True, explore=explore)
    num_legal = int(step.legal_actions_mask[0].sum())      # 20 in the opening
    assert abs(steps[0].old_log_prob[0].item() - math.log(1 / num_legal)) < 0.2
    assert steps[0].old_log_prob[1].item() > math.log(0.5)  # untouched board: log pi(a) of the peaky move
    assert steps[0].noisy.tolist() == [True, False]
    s = m.episode_summary()
    assert s["explore_moves"] == 1 and s["explore_offprior_share"] in (0.0, 1.0)
