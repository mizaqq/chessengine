import torch
from src.core.types import StepRecord

WHITE = 1  # OpenSpiel convention: player 1 = white
BLACK = 0


def compute_returns_for_model(
    steps: list[StepRecord],
    model_id: int,
    bootstrap_value: torch.Tensor,
    gamma: float,
) -> list[torch.Tensor]:
    """Compute discounted returns for a specific model.

    Handles cross-player done propagation: if the game ends on the
    opponent's step, the return chain is still cut correctly.

    Args:
        steps: List of StepRecord from rollout collection.
        model_id: 1 for white, 0 for black (OpenSpiel convention).
        bootstrap_value: [num_envs] value estimate for bootstrapping.
        gamma: Discount factor.

    Returns:
        List of [num_envs] tensors, one per step.
    """
    R = bootstrap_value.clone()
    returns = [None] * len(steps)

    for i in reversed(range(len(steps))):
        step = steps[i]
        is_mine = (step.player == model_id)
        done_f = step.done.float()

        terminal_r = step.terminal_r_white if model_id == WHITE else step.terminal_r_black
        R = R * (1.0 - done_f) + terminal_r

        reward = step.reward_white if model_id == WHITE else -step.reward_white
        new_R = reward + gamma * R
        R = torch.where(is_mine, new_R, R)

        returns[i] = R.clone()

    return returns
