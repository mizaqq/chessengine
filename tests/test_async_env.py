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
