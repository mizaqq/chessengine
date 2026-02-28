import torch
import numpy as np
from src.core.interfaces import VectorEnv
from src.core.types import EnvStep
from src.envs.open_spiel_env import OpenSpielEnv


class OpenSpielVectorEnv(VectorEnv):
    """Vectorized OpenSpiel environment adapter implementing VectorEnv interface."""
    
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [OpenSpielEnv() for _ in range(num_envs)]

    def reset(self) -> EnvStep:
        """Reset all environments and return initial step."""
        for env in self.envs:
            env.reset()
        
        obs = torch.tensor(
            np.array([env.state() for env in self.envs], dtype=np.float32)
        )
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])
        reward = torch.zeros(self.num_envs)
        done = torch.tensor([env.is_done() for env in self.envs], dtype=torch.bool)
        info = {}
        
        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            reward=reward,
            done=done,
            info=info
        )

    def step(self, actions):
        """Execute actions in all environments and return step results."""
        for env, action in zip(self.envs, actions):
            if not env.is_done():
                env.step(int(action))
        
        obs = torch.tensor(
            np.array([env.state() for env in self.envs], dtype=np.float32)
        )
        legal_actions_mask = torch.stack([env.get_legal_actions() for env in self.envs])
        reward = torch.zeros(self.num_envs)
        done = torch.tensor([env.is_done() for env in self.envs], dtype=torch.bool)
        info = {}
        
        return EnvStep(
            obs=obs,
            legal_actions_mask=legal_actions_mask,
            reward=reward,
            done=done,
            info=info
        )
