"""PPO update step (openspec/changes/add-ppo-gae/specs/policy-update/spec.md).

The maths is tested on a network in eval mode so BatchNorm uses fixed statistics;
in training mode the re-evaluated probabilities drift from the sampled ones (the
documented BatchNorm deviation).
"""
import torch
from torch.distributions import Categorical
from torch.testing import assert_close

from src.envs.open_spiel_vector_env import OpenSpielVectorEnv
from src.model.chess_model import ChessPolicyProbs
from src.model.training import (
    A2C_PRESET, PPO_PRESET, _collect_rollout, _concat_batches, _gather_side,
    normalized_entropy, ppo_policy_loss, refresh_norm_stats, run_chess_training, update_model, WHITE, BLACK,
)
from src.model.returns import compute_gae_for_model
from src.training.metrics import MetricsAggregator


class CountingAdam(torch.optim.Adam):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def _batch(model, num_envs=3, num_steps=4, seed=0):
    torch.manual_seed(seed)
    envs = OpenSpielVectorEnv(num_envs)
    metrics = MetricsAggregator()
    steps, env_step = _collect_rollout(
        envs, model, model, envs.reset(), num_steps, num_envs,
        {"win": 2.0, "loss": -2.0, "draw": -0.5}, metrics, oriented=True,
    )
    boot = torch.zeros(num_envs)
    gs = []
    for side in (WHITE, BLACK):
        adv, tgt = compute_gae_for_model(steps, side, boot, 0.99, 0.95, env_step.material, 0.2)
        gs.append(_gather_side(steps, adv, tgt, side))
    return steps, _concat_batches(gs)


def test_rollout_stores_detached_tensors_and_leaves_no_grad():
    model = ChessPolicyProbs(num_filters=8)
    steps, batch = _batch(model)
    for s in steps:
        for name in ("obs", "old_log_prob", "old_value", "entropy"):
            assert not getattr(s, name).requires_grad
    assert all(p.grad is None for p in model.parameters())
    assert batch["obs"].shape[0] == 12  # 3 envs x 4 steps, both colours


def test_first_minibatch_ratio_is_one_and_loss_is_minus_mean_adv():
    model = ChessPolicyProbs(num_filters=8).eval()
    _, batch = _batch(model)
    with torch.no_grad():
        probs, _ = model(batch["obs"], batch["legal_mask"])
        new_lp = Categorical(probs=probs).log_prob(batch["action"])
    loss, ratio = ppo_policy_loss(new_lp, batch["old_log_prob"], batch["adv"], clip_epsilon=0.2)
    assert_close(ratio, torch.ones_like(ratio), atol=1e-5, rtol=0)
    assert_close(loss, -batch["adv"].mean(), atol=1e-5, rtol=0)
    assert ((ratio - 1).abs() > 0.2).float().mean() == 0


def test_positive_advantage_above_clip_range_has_no_gradient():
    new_lp = torch.log(torch.tensor([0.3]), ).requires_grad_(True)
    loss, ratio = ppo_policy_loss(new_lp, torch.log(torch.tensor([0.2])), torch.tensor([2.0]), 0.2)
    assert_close(ratio, torch.tensor([1.5]))
    assert_close(loss, torch.tensor(-2.4))   # min(1.5*2, 1.2*2)
    loss.backward()
    assert_close(new_lp.grad, torch.tensor([0.0]))


def test_negative_advantage_below_clip_range_has_no_gradient():
    new_lp = torch.log(torch.tensor([0.1])).requires_grad_(True)
    loss, ratio = ppo_policy_loss(new_lp, torch.log(torch.tensor([0.2])), torch.tensor([-1.0]), 0.2)
    assert_close(ratio, torch.tensor([0.5]))
    assert_close(loss, torch.tensor(0.8))    # -min(0.5*-1, 0.8*-1) = -(-0.8)
    loss.backward()
    assert_close(new_lp.grad, torch.tensor([0.0]))


def test_unclipped_sample_inside_range_keeps_gradient():
    new_lp = torch.log(torch.tensor([0.22])).requires_grad_(True)
    loss, _ = ppo_policy_loss(new_lp, torch.log(torch.tensor([0.2])), torch.tensor([2.0]), 0.2)
    loss.backward()
    assert new_lp.grad.abs().item() > 0


def test_epochs_times_minibatches_optimizer_steps_and_sizes():
    model = ChessPolicyProbs(num_filters=8)
    _, batch = _batch(model)   # 12 samples
    opt = CountingAdam(model.parameters(), lr=1e-4)
    out = update_model(model, opt, batch, epochs=4, minibatches=4, clip_epsilon=0.2,
                       normalize_advantage=True, value_coef=0.5)
    assert opt.steps == 16 and out["optimizer_steps"] == 16 and out["sample_count"] == 12
    assert out["clip_fraction"] is not None and out["approx_kl"] is not None
    assert not any(torch.isnan(torch.tensor(v)) for v in (out["total_loss"], out["approx_kl"]))


def test_more_minibatches_than_samples_uses_single_sample_minibatches_without_nan():
    model = ChessPolicyProbs(num_filters=8)
    _, batch = _batch(model, num_envs=1, num_steps=2)   # 2 samples
    opt = CountingAdam(model.parameters(), lr=1e-4)
    out = update_model(model, opt, batch, epochs=1, minibatches=8, clip_epsilon=0.2, normalize_advantage=True)
    assert opt.steps == 2
    assert out["policy_loss"] == 0.0        # normalised single-sample advantage is 0
    assert not torch.isnan(torch.tensor(out["total_loss"]))


def test_empty_batch_takes_no_step():
    model = ChessPolicyProbs(num_filters=8)
    opt = CountingAdam(model.parameters())
    out = update_model(model, opt, None, **{k: v for k, v in PPO_PRESET.items() if k != "gae_lambda"})
    assert opt.steps == 0 and out["optimizer_steps"] == 0 and out["sample_count"] == 0


def test_value_coef_doubling_doubles_value_head_gradient():
    torch.manual_seed(1)
    model = ChessPolicyProbs(num_filters=8).eval()
    _, batch = _batch(model)

    def value_head_grad(value_coef):
        # a frozen optimizer so parameters do not move between the two calls
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        update_model(model, opt, batch, epochs=1, minibatches=1, clip_epsilon=None,
                     value_coef=value_coef, entropy_coef=0.0, grad_clip=1e9)
        return torch.cat([p.grad.flatten().clone() for p in model.value_head[-1].parameters()])

    g1, g2 = value_head_grad(1.0), value_head_grad(2.0)
    assert_close(g2, 2 * g1, atol=1e-6, rtol=1e-4)


def test_a2c_preset_equals_reinforce_with_baseline_gradient():
    torch.manual_seed(2)
    model = ChessPolicyProbs(num_filters=8).eval()
    _, batch = _batch(model)
    coef = 0.01

    # direct: -(logp * A).mean() + (v - target)^2.mean() - coef * norm_ent.mean()
    model.zero_grad()
    probs, v = model(batch["obs"], batch["legal_mask"])
    dist = Categorical(probs=probs)
    ent_norm = normalized_entropy(dist.entropy(), batch["num_legal"])
    direct = (-(dist.log_prob(batch["action"]) * batch["adv"]).mean()
              + (v.squeeze(-1) - batch["target"]).pow(2).mean() - coef * ent_norm.mean())
    direct.backward()
    expected = [p.grad.clone() for p in model.parameters()]

    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    update_model(model, opt, batch, **{k: v for k, v in A2C_PRESET.items() if k != "gae_lambda"},
                 entropy_coef=coef, grad_clip=1e9)
    for e, p in zip(expected, model.parameters()):
        assert_close(p.grad, e, atol=1e-5, rtol=1e-4)


def test_training_keeps_model_in_eval_mode_and_refreshes_bn_stats():
    model = ChessPolicyProbs(num_filters=8)
    bn = model.res_tower[0].bn1
    before = bn.running_mean.clone()
    run_chess_training(OpenSpielVectorEnv(2), model, None, torch.optim.Adam(model.parameters()), None,
                       episodes=1, steps=3, log_interval=1)
    assert not model.training
    assert not torch.allclose(bn.running_mean, before)   # statistics were refreshed from the rollout


def test_refresh_norm_stats_returns_eval_mode_and_first_ratio_is_one_in_loop():
    """After a refresh the collection and the update see the same statistics."""
    model = ChessPolicyProbs(num_filters=8)
    _, batch = _batch(model.eval())
    refresh_norm_stats(model, batch["obs"], batch["legal_mask"])
    assert not model.training
    _, batch2 = _batch(model)   # sampled with the refreshed statistics
    with torch.no_grad():
        probs, _ = model(batch2["obs"], batch2["legal_mask"])
        lp = Categorical(probs=probs).log_prob(batch2["action"])
    assert_close(torch.exp(lp - batch2["old_log_prob"]), torch.ones_like(lp), atol=1e-5, rtol=0)


def test_forced_move_has_zero_normalized_entropy():
    ent = torch.tensor([1.19e-7, 0.5, 2.0])
    n = torch.tensor([1.0, 2.0, 20.0])
    out = normalized_entropy(ent, n)
    assert out[0] == 0.0
    assert_close(out[1], torch.tensor(0.5 / torch.log(torch.tensor(2.0)).item()))
    assert (out <= 1.0 + 1e-6).all()
