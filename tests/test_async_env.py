import pytest
import torch
from src.environment.async_vector_env import AsyncVectorEnv


def test_async_env_initialization():
    num_envs = 2
    env = AsyncVectorEnv(num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()


def test_async_env_reset():
    env = AsyncVectorEnv(num_envs=2)
    obs, legal_actions = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (2, 20, 8, 8)
    assert isinstance(legal_actions, torch.Tensor)
    assert legal_actions.shape == (2, 4674)
    env.close()


def test_async_env_step():
    env = AsyncVectorEnv(num_envs=2)
    obs, legal = env.reset()

    actions = []
    for i in range(2):
        legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
        actions.append(legal_indices[0].item())

    actions_tensor = torch.tensor(actions)
    next_obs, rewards, dones, infos, next_legal = env.step(actions_tensor)

    assert next_obs.shape == (2, 20, 8, 8)
    assert rewards.shape == (2, 1)
    assert dones.shape == (2, 1)
    assert len(infos) == 2
    assert next_legal.shape == (2, 4674)
    env.close()


def test_async_env_reward_calculation():
    """Rewards should reflect piece-difference changes, not always be zero."""
    import random
    env = AsyncVectorEnv(num_envs=2)
    obs, legal = env.reset()

    all_rewards = []
    for _ in range(200):
        actions = []
        for i in range(2):
            legal_indices = (legal[i] == 1).nonzero(as_tuple=True)[0]
            actions.append(legal_indices[random.randint(0, len(legal_indices) - 1)].item())
        actions_tensor = torch.tensor(actions)
        obs, rewards, dones, infos, legal = env.step(actions_tensor)
        all_rewards.append(rewards)

    all_rewards = torch.cat(all_rewards, dim=0)
    assert not torch.all(all_rewards == 0.0), "Rewards should not all be zero after 200 random steps"
    env.close()
