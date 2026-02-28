import torch
from typing import List, Tuple
from torch.distributions import Categorical


class RolloutCollector:
    """Collects rollout data during training episodes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all collected data."""
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []
        self.states = []
        self.actions = []
    
    def add_step(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        state: torch.Tensor = None,
        action: torch.Tensor = None,
    ):
        """Add data from a single step."""
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        if state is not None:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
    
    def get_data(self) -> Tuple[List, List, List, List]:
        """Return collected rollout data."""
        return self.rewards, self.values, self.log_probs, self.entropies
    
    def __len__(self):
        return len(self.rewards)
