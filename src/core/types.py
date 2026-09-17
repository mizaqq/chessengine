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
    """One rollout step for all boards. Plain tensors, no autograd graph: the
    update re-runs the network on `obs` (PPO needs several passes per rollout)."""
    player: Tensor             # [num_envs], 1=white, 0=black (OpenSpiel convention)
    obs: Tensor                # [num_envs, 20, 8, 8], as the acting model saw it (oriented if shared)
    legal_mask: Tensor         # [num_envs, 4674]
    action: Tensor             # [num_envs], sampled action
    old_log_prob: Tensor       # [num_envs], log pi_old(action) at collection
    old_value: Tensor          # [num_envs], V_old(obs) at collection
    entropy: Tensor            # [num_envs], policy entropy at collection (logging only)
    num_legal: Tensor          # [num_envs], count of legal moves
    potential_white: Tensor    # [num_envs], white-view material of the position the mover saw
    done: Tensor               # [num_envs], bool
    terminal_r_white: Tensor   # [num_envs], precomputed terminal reward for white
    terminal_r_black: Tensor   # [num_envs], precomputed terminal reward for black
    noisy: Tensor = None       # [num_envs] bool, move drawn from the exploration mixture (diagnostics only)
    learner: Tensor = None     # [num_envs] bool, ply chosen by the learner (False = frozen pool opponent; dropped from the update)
