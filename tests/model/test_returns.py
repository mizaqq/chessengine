import torch
from torch.testing import assert_close
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model

WHITE = 1  # OpenSpiel convention
BLACK = 0


def _make_step(player, reward_white, done, terminal_r_w=0.0, terminal_r_b=0.0, n=1):
    return StepRecord(
        player=torch.tensor([player] * n, dtype=torch.long),
        value_white=torch.zeros(n, 1),
        value_black=torch.zeros(n, 1),
        log_prob_white=torch.zeros(n),
        log_prob_black=torch.zeros(n),
        entropy_white=torch.zeros(n),
        entropy_black=torch.zeros(n),
        reward_white=torch.tensor([reward_white] * n),
        done=torch.tensor([done] * n),
        terminal_r_white=torch.tensor([terminal_r_w] * n),
        terminal_r_black=torch.tensor([terminal_r_b] * n),
    )


def test_returns_single_player_no_done():
    """All white steps, no done. Standard discounted returns."""
    steps = [
        _make_step(player=WHITE, reward_white=1.0, done=False),
        _make_step(player=WHITE, reward_white=2.0, done=False),
        _make_step(player=WHITE, reward_white=3.0, done=False),
    ]
    bootstrap = torch.tensor([10.0])
    returns = compute_returns_for_model(steps, model_id=WHITE, bootstrap_value=bootstrap, gamma=0.99)
    # R_2 = 3 + 0.99 * 10 = 12.9
    # R_1 = 2 + 0.99 * 12.9 = 14.771
    # R_0 = 1 + 0.99 * 14.771 = 15.62329
    assert_close(returns[2], torch.tensor([12.9]))
    assert_close(returns[1], torch.tensor([14.771]))
    assert_close(returns[0], torch.tensor([15.62329]))


def test_returns_alternating_no_done():
    """White and black alternate. White's return skips black's steps."""
    steps = [
        _make_step(player=WHITE, reward_white=1.0, done=False),
        _make_step(player=BLACK, reward_white=-2.0, done=False),
        _make_step(player=WHITE, reward_white=3.0, done=False),
    ]
    bootstrap = torch.tensor([0.0])
    returns = compute_returns_for_model(steps, model_id=WHITE, bootstrap_value=bootstrap, gamma=1.0)
    # gamma=1.0 for simplicity. White reward = reward_white as-is.
    # Step 2 (white): R = 3 + 1.0 * 0 = 3
    # Step 1 (black): is_mine=False, R stays 3
    # Step 0 (white): R = 1 + 1.0 * 3 = 4
    assert_close(returns[2], torch.tensor([3.0]))
    assert_close(returns[0], torch.tensor([4.0]))


def test_returns_done_on_own_step():
    """Done on white's step cuts the chain."""
    steps = [
        _make_step(player=WHITE, reward_white=1.0, done=False),
        _make_step(player=WHITE, reward_white=5.0, done=True, terminal_r_w=2.0),
        _make_step(player=WHITE, reward_white=0.0, done=False),  # new game
    ]
    bootstrap = torch.tensor([10.0])
    returns = compute_returns_for_model(steps, model_id=WHITE, bootstrap_value=bootstrap, gamma=1.0)
    # Step 2 (new game): R = 0 + 1.0 * 10 = 10
    # Step 1 (done=T):
    #   A: R = 10 * (1-1) + 2.0 = 2.0  (chain cut, terminal reward injected)
    #   B: is_mine=True → R = 5 + 1.0 * 2.0 = 7.0
    # Step 0: R = 1 + 1.0 * 7.0 = 8.0
    assert_close(returns[2], torch.tensor([10.0]))
    assert_close(returns[1], torch.tensor([7.0]))
    assert_close(returns[0], torch.tensor([8.0]))


def test_returns_cross_player_done():
    """Black checkmates at step 1. White's return at step 0 must NOT include step 2 (new game)."""
    steps = [
        _make_step(player=WHITE, reward_white=0.0, done=False),
        _make_step(player=BLACK, reward_white=0.0, done=True, terminal_r_w=-2.0, terminal_r_b=2.0),
        _make_step(player=WHITE, reward_white=0.0, done=False),  # new game after reset
    ]
    bootstrap = torch.tensor([5.0])
    returns = compute_returns_for_model(steps, model_id=WHITE, bootstrap_value=bootstrap, gamma=1.0)
    # Step 2 (white, new game): R = 0 + 1.0 * 5.0 = 5.0
    # Step 1 (black, done=T):
    #   A: R = 5.0 * (1-1) + (-2.0) = -2.0  (chain cut!)
    #   B: is_mine=False → R stays -2.0
    # Step 0 (white): R = 0 + 1.0 * (-2.0) = -2.0
    assert_close(returns[2], torch.tensor([5.0]))
    assert_close(returns[0], torch.tensor([-2.0]))


def test_returns_black_model_perspective():
    """Rewards are flipped for black model."""
    steps = [
        _make_step(player=WHITE, reward_white=1.0, done=False),
        _make_step(player=BLACK, reward_white=-3.0, done=False),  # from white persp; black gets +3
    ]
    bootstrap = torch.tensor([0.0])
    returns = compute_returns_for_model(steps, model_id=BLACK, bootstrap_value=bootstrap, gamma=1.0)
    # For black (model_id=0): reward = -reward_white = -(-3.0) = +3.0
    # Step 1 (black): R = 3.0 + 0 = 3.0
    # Step 0 (white): is_mine=False → R stays 3.0
    assert_close(returns[1], torch.tensor([3.0]))


def test_returns_multi_env():
    """Two envs with different players at the same step."""
    steps = [
        StepRecord(
            player=torch.tensor([WHITE, BLACK]),
            value_white=torch.zeros(2, 1),
            value_black=torch.zeros(2, 1),
            log_prob_white=torch.zeros(2),
            log_prob_black=torch.zeros(2),
            entropy_white=torch.zeros(2),
            entropy_black=torch.zeros(2),
            reward_white=torch.tensor([1.0, -1.0]),
            done=torch.tensor([False, False]),
            terminal_r_white=torch.zeros(2),
            terminal_r_black=torch.zeros(2),
        ),
    ]
    bootstrap = torch.tensor([0.0, 0.0])
    returns_w = compute_returns_for_model(steps, model_id=WHITE, bootstrap_value=bootstrap, gamma=1.0)
    # Env 0: white acted, reward_white=1.0 → R = 1 + 0 = 1.0
    # Env 1: black acted, is_mine=False → R stays 0.0
    assert_close(returns_w[0], torch.tensor([1.0, 0.0]))

    returns_b = compute_returns_for_model(steps, model_id=BLACK, bootstrap_value=bootstrap, gamma=1.0)
    # Env 0: white acted, is_mine=False → R stays 0.0
    # Env 1: black acted, reward_for_black = -(-1.0) = 1.0 → R = 1.0
    assert_close(returns_b[0], torch.tensor([0.0, 1.0]))
