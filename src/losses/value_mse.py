import torch
from torch.nn.functional import mse_loss
from typing import List


def compute_value_loss(
    values: List[torch.Tensor],
    returns: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute value function MSE loss (critic loss).
    
    Args:
        values: List of value predictions for each step
        returns: List of return targets for each step
    
    Returns:
        Mean squared error between values and returns
    """
    loss = 0
    for value, target in zip(values, returns):
        loss += mse_loss(value, target.detach())
    return loss.mean()
