from dataclasses import dataclass
from typing import Any, Dict
import torch

Tensor = torch.Tensor


@dataclass
class EnvStep:
    obs: Tensor
    legal_actions_mask: Tensor
    material: Tensor        # [num_envs], white-view material balance of `obs`
    done: Tensor
    current_player: Tensor  # [num_envs], 1=white, 0=black (OpenSpiel convention)
    info: Dict[str, Any]


@dataclass
class StepRecord:
    player: Tensor             # [num_envs], 1=white, 0=black (OpenSpiel convention)
    value_white: Tensor        # [num_envs, 1], grad-connected for white model only
    value_black: Tensor        # [num_envs, 1], grad-connected for black model only
    log_prob_white: Tensor     # [num_envs], grad-connected for white model only
    log_prob_black: Tensor     # [num_envs], grad-connected for black model only
    entropy_white: Tensor      # [num_envs], grad-connected for white model only
    entropy_black: Tensor      # [num_envs], grad-connected for black model only
    num_legal_white: Tensor   # [num_envs], count of legal moves for white
    num_legal_black: Tensor   # [num_envs], count of legal moves for black
    potential_white: Tensor    # [num_envs], white-view material of the position the mover saw
    done: Tensor               # [num_envs], bool
    terminal_r_white: Tensor   # [num_envs], precomputed terminal reward for white
    terminal_r_black: Tensor   # [num_envs], precomputed terminal reward for black
