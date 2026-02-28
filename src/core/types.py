from dataclasses import dataclass
from typing import Any, Dict
import torch

Tensor = torch.Tensor


@dataclass
class EnvStep:
    obs: Tensor
    legal_actions_mask: Tensor
    reward: Tensor
    done: Tensor
    info: Dict[str, Any]


@dataclass
class RolloutBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    legal_actions_mask: Tensor
