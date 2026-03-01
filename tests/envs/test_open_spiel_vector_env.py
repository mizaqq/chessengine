import torch
import random
import pytest
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv


def test_vector_env_reset_returns_current_player():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.current_player.shape == (2,)
    # OpenSpiel chess 1.6.x has player 1 (black) to move first after reset
    assert (step.current_player == 1).all(), "After reset all envs should have same current_player"


def test_vector_env_step_returns_current_player():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    legal = step.legal_actions_mask
    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.current_player.shape == (2,)
    # After first move (by player 1), current_player alternates to 0
    assert (step.current_player == 0).all(), "After first move current_player should alternate"


def test_vector_env_reset_returns_valid_shapes():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.legal_actions_mask.shape == (2, 4674)
    assert step.done.dtype == torch.bool
    assert step.reward.shape == (2,)


def test_vector_env_step_returns_env_step():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    legal = step.legal_actions_mask
    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.reward.shape == (2,)
    assert step.done.shape == (2,)
    assert isinstance(step.info, dict)


def test_vector_env_auto_reset():
    env = OpenSpielVectorEnv(num_envs=1)
    step = env.reset()
    for _ in range(500):
        legal = step.legal_actions_mask
        legal_indices = (legal[0] == 1).nonzero(as_tuple=True)[0]
        action = legal_indices[random.randint(0, len(legal_indices) - 1)].item()
        step = env.step(torch.tensor([action]))
        if "game_results" in step.info:
            assert step.info["game_results"][0] in ("white_win", "black_win", "draw")
            assert 0 in step.info["terminal_observations"]
            break
    else:
        pytest.fail("Game never ended in 500 steps")
