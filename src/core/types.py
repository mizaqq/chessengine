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
    current_player: Tensor  # [num_envs], 1=white, 0=black (OpenSpiel convention)
    info: Dict[str, Any]


@dataclass
class StepRecord:
    player: Tensor           # [num_envs], 1=white, 0=black (OpenSpiel convention)
    value: Tensor            # [num_envs, 1]
    log_prob: Tensor         # [num_envs, 1]
    entropy: Tensor          # [num_envs, 1]
    reward_white: Tensor     # [num_envs], piece-diff from white's perspective
    done: Tensor             # [num_envs], bool
    terminal_r_white: Tensor # [num_envs], precomputed terminal reward for white
    terminal_r_black: Tensor # [num_envs], precomputed terminal reward for black


@dataclass
class RolloutBatch:
    obs: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    legal_actions_mask: Tensor
