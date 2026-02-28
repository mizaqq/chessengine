import torch
from src.envs.open_spiel_vector_env import OpenSpielVectorEnv


def test_vector_env_reset_returns_valid_shapes():
    env = OpenSpielVectorEnv(num_envs=2)
    step = env.reset()
    assert step.obs.shape[0] == 2
    assert step.legal_actions_mask.shape == (2, 4674)
    assert step.done.dtype == torch.bool
