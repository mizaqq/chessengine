import torch
import random
import pytest
from src.envs.open_spiel_async_vector_env import OpenSpielAsyncVectorEnv


def test_async_env_reset():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.legal_actions_mask.shape == (2, 4674)
    assert step.reward.shape == (2,)
    assert step.done.shape == (2,)
    env.close()


def test_async_env_step():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    step = env.reset()
    actions = []
    for i in range(2):
        legal_indices = (step.legal_actions_mask[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())
    step = env.step(torch.tensor(actions))
    assert step.obs.shape == (2, 20, 8, 8)
    assert step.reward.shape == (2,)
    assert step.done.shape == (2,)
    assert isinstance(step.info, dict)
    env.close()


def test_async_env_reset_returns_current_player():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    try:
        step = env.reset()
        assert step.current_player.shape == (2,)
        # OpenSpiel chess: player 1 = white moves first after reset
        assert (step.current_player == 1).all(), (
            "After reset all envs should be white-to-move (player 1)"
        )
    finally:
        env.close()


def test_async_env_step_returns_current_player():
    env = OpenSpielAsyncVectorEnv(num_envs=2)
    try:
        step = env.reset()
        legal = step.legal_actions_mask
        actions = []
        for i in range(2):
            legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
            actions.append(legal_indices[0].item())
        step = env.step(torch.tensor(actions))
        assert step.current_player.shape == (2,)
        # After white (player 1) moves, current_player should be 0 (black)
        assert (step.current_player == 0).all(), (
            "After white moves, should be black-to-move (player 0)"
        )
    finally:
        env.close()


def test_async_env_auto_reset_with_game_result():
    env = OpenSpielAsyncVectorEnv(num_envs=1)
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
    env.close()
