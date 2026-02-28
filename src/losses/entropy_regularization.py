import torch
from typing import List


def compute_entropy_bonus(
    entropies: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute entropy bonus for exploration.
    
    Args:
        entropies: List of entropy values for each step
    
    Returns:
        Mean entropy across all steps
    """
    if len(entropies) == 0:
        return torch.tensor(0.0)
    return torch.stack(entropies).mean()
