from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict
import torch


class VectorEnv(ABC):
    """Abstract base class for vectorized environments."""

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset all environments and return initial observations.
        Returns:
            observations (torch.Tensor): Batch of initial observations. Shape: (num_envs, *obs_shape)
            legal_actions (torch.Tensor): Batch of legal action masks. Shape: (num_envs, num_actions)
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]], torch.Tensor]:
        """
        Step all environments with the given actions.
        Args:
            actions (torch.Tensor): Batch of actions. Shape: (num_envs, *action_shape)
        Returns:
            Tuple containing:
                - next_states (torch.Tensor): Batch of next observations.
                - rewards (torch.Tensor): Batch of rewards.
                - dones (torch.Tensor): Batch of done flags.
                - infos (List[Dict[str, Any]]): Auxiliary information (including terminal observations).
                - legal_actions (torch.Tensor): Batch of legal action masks.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
