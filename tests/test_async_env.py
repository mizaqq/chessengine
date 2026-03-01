import pytest
import torch
from src.environment.async_vector_env import AsyncVectorEnv


def test_async_env_initialization():
    num_envs = 2
    env = AsyncVectorEnv(num_envs=num_envs)
    assert env.num_envs == num_envs
    env.close()
