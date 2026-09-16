"""GAE(gamma, lambda) advantages mirroring the shaped returns walk.

Scenarios from openspec/changes/add-ppo-gae/specs/advantage-estimation/spec.md.
Shaped reward per own step: scale * (gamma * Phi(next own position) - Phi(this)).
With all potentials 0 and scale 0, the tests below set the reward through the
potentials (potential_white = -reward under gamma=1 telescopes to reward_i =
Phi_{i+1} - Phi_i), or use zero reward and a bootstrap.
"""
import torch
from torch.testing import assert_close

from src.core.types import StepRecord
from src.model.returns import compute_gae_for_model, compute_returns_for_model

WHITE, BLACK = 1, 0


def _step(player, value=0.0, potential_white=0.0, done=False, tr_w=0.0, tr_b=0.0, n=1):
    return StepRecord(
        player=torch.tensor([player] * n, dtype=torch.long),
        obs=torch.zeros(n, 20, 8, 8),
        legal_mask=torch.ones(n, 4674),
        action=torch.zeros(n, dtype=torch.long),
        old_log_prob=torch.zeros(n),
        old_value=torch.tensor([value] * n, dtype=torch.float32),
        entropy=torch.zeros(n),
        num_legal=torch.ones(n) * 20,
        potential_white=torch.tensor([potential_white] * n, dtype=torch.float32),
        done=torch.tensor([done] * n),
        terminal_r_white=torch.tensor([tr_w] * n),
        terminal_r_black=torch.tensor([tr_b] * n),
    )


def _gae(steps, model_id, bootstrap, gamma, lam, final_material=0.0, scale=1.0):
    return compute_gae_for_model(
        steps, model_id=model_id, bootstrap_value=torch.tensor([bootstrap]), gamma=gamma,
        lam=lam, final_material=torch.tensor([final_material]), shaping_scale=scale,
    )


def _three_white_steps_rewards_1_2_3():
    """gamma-independent trick: rewards 1, 2, 3 come from potentials with scale 1 and
    the shaping formula; simpler to encode them as potentials under gamma = 0.99 is
    messy, so use scale 0 and a hand-built reward path via terminal rewards is
    impossible for non-terminal steps. Instead: encode reward via potentials with
    gamma folded in is spec-specific; here we emulate the spec numerically by
    computing the expected values with the same formula the spec uses."""
    # potentials chosen so that shaped = gamma*phi_next - phi equals 1, 2, 3 at gamma=0.99
    # phi_3 (final) = 0 -> shaped_2 = -phi_2 = 3 -> phi_2 = -3
    # shaped_1 = 0.99*phi_2 - phi_1 = 2 -> phi_1 = -2.97 - 2 = -4.97
    # shaped_0 = 0.99*phi_1 - phi_0 = 1 -> phi_0 = -4.9203 - 1 = -5.9203
    return [
        _step(WHITE, value=0.5, potential_white=-5.9203),
        _step(WHITE, value=0.5, potential_white=-4.97),
        _step(WHITE, value=0.5, potential_white=-3.0),
    ]


def test_lambda_one_is_return_minus_value():
    steps = _three_white_steps_rewards_1_2_3()
    adv, target = _gae(steps, WHITE, bootstrap=10.0, gamma=0.99, lam=1.0)
    rets = compute_returns_for_model(steps, WHITE, torch.tensor([10.0]), 0.99, torch.tensor([0.0]), 1.0)
    for i, expected in enumerate([15.62329, 14.771, 12.9]):
        assert_close(rets[i], torch.tensor([expected]), atol=1e-3, rtol=0)
        assert_close(target[i], rets[i], atol=1e-5, rtol=0)
        assert_close(adv[i], rets[i] - 0.5, atol=1e-5, rtol=0)


def test_lambda_zero_is_one_step_residual():
    steps = _three_white_steps_rewards_1_2_3()
    adv, _ = _gae(steps, WHITE, bootstrap=10.0, gamma=0.99, lam=0.0)
    # delta_0 = 1 + 0.99*0.5 - 0.5 = 0.995 ; delta_1 = 2 + 0.495 - 0.5 = 1.995 ; delta_2 = 3 + 9.9 - 0.5 = 12.4
    assert_close(torch.cat(adv), torch.tensor([0.995, 1.995, 12.4]), atol=1e-3, rtol=0)


def test_intermediate_lambda_blends():
    steps = _three_white_steps_rewards_1_2_3()
    adv, _ = _gae(steps, WHITE, bootstrap=10.0, gamma=0.99, lam=0.95)
    a2 = 12.4
    a1 = 1.995 + 0.99 * 0.95 * a2
    a0 = 0.995 + 0.99 * 0.95 * a1
    assert_close(torch.cat(adv), torch.tensor([a0, a1, a2]), atol=1e-3, rtol=0)


def test_opponent_steps_are_skipped_without_discount():
    # white, black, white; white rewards 1 (step 0) and 3 (step 2) via potentials,
    # gamma 1, scale 1: shaped_2 = phi_final - phi_2 = 0 - (-3) = 3 ; shaped_0 = phi_2 - phi_0 = -3 - (-4) = 1
    steps = [_step(WHITE, potential_white=-4.0), _step(BLACK, potential_white=-4.0), _step(WHITE, potential_white=-3.0)]
    adv, _ = _gae(steps, WHITE, bootstrap=0.0, gamma=1.0, lam=1.0)
    assert_close(adv[0], torch.tensor([4.0]))
    assert_close(adv[2], torch.tensor([3.0]))
    assert_close(adv[1], torch.tensor([0.0]))  # no advantage produced at the black step


def test_game_ends_on_opponent_move_uses_terminal_reward_not_bootstrap():
    steps = [_step(WHITE, value=0.2), _step(BLACK, done=True, tr_w=2.0, tr_b=-2.0)]
    adv, target = _gae(steps, WHITE, bootstrap=5.0, gamma=0.99, lam=0.95)
    assert_close(adv[0], torch.tensor([0.99 * 2.0 - 0.2]), atol=1e-6, rtol=0)
    assert_close(target[0], torch.tensor([0.99 * 2.0]), atol=1e-6, rtol=0)


def test_new_game_after_done_is_not_linked_to_old_one():
    steps = [_step(WHITE, value=0.0), _step(WHITE, value=0.0, done=True, tr_w=2.0), _step(WHITE, value=0.0)]
    adv_a, _ = _gae(steps, WHITE, bootstrap=7.0, gamma=0.99, lam=0.95)
    steps[2] = _step(WHITE, value=0.0)
    adv_b, _ = _gae(steps, WHITE, bootstrap=-7.0, gamma=0.99, lam=0.95)
    # steps before the done ignore the bootstrap of the next game
    assert_close(adv_a[0], adv_b[0])
    assert_close(adv_a[1], adv_b[1])
    # the new game sees only its own bootstrap
    assert_close(adv_a[2], torch.tensor([0.99 * 7.0]))
    assert_close(adv_b[2], torch.tensor([-0.99 * 7.0]))


def test_lambda_one_matches_returns_on_random_alternating_rollout():
    torch.manual_seed(0)
    n = 5
    steps = []
    for t in range(9):
        steps.append(StepRecord(
            player=torch.randint(0, 2, (n,)),
            obs=torch.zeros(n, 20, 8, 8), legal_mask=torch.ones(n, 4674),
            action=torch.zeros(n, dtype=torch.long), old_log_prob=torch.zeros(n),
            old_value=torch.randn(n), entropy=torch.zeros(n), num_legal=torch.ones(n) * 20,
            potential_white=torch.randn(n), done=torch.rand(n) < 0.2,
            terminal_r_white=torch.randn(n), terminal_r_black=torch.randn(n),
        ))
    boot, final = torch.randn(n), torch.randn(n)
    for side in (WHITE, BLACK):
        rets = compute_returns_for_model(steps, side, boot, 0.97, final, 0.2)
        adv, target = compute_gae_for_model(steps, side, boot, 0.97, 1.0, final, 0.2)
        for i, s in enumerate(steps):
            mine = s.player == side
            assert_close(target[i][mine], rets[i][mine], atol=1e-5, rtol=0)
            assert_close(adv[i][mine], (rets[i] - s.old_value)[mine], atol=1e-5, rtol=0)
