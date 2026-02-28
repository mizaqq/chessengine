import torch
from typing import List


def compute_policy_gradient_loss(
    log_probs: List[torch.Tensor],
    advantages: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute policy gradient loss (actor loss).
    
    Args:
        log_probs: List of log probabilities for each step
        advantages: List of advantage values for each step
    
    Returns:
        Policy gradient loss (negative for gradient ascent)
    """
    loss = 0
    for log_prob, advantage in zip(log_probs, advantages):
        loss -= log_prob * advantage.detach()
    return loss.mean()
