import torch
from typing import Dict


class ComposedLoss:
    """Composes individual loss components into total training loss."""
    
    def compute(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_bonus: torch.Tensor,
        entropy_coef: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from individual components.
        
        Args:
            policy_loss: Policy gradient loss
            value_loss: Value function MSE loss
            entropy_bonus: Entropy bonus for exploration
            entropy_coef: Coefficient for entropy regularization
        
        Returns:
            Dictionary with individual losses and total loss
        """
        total_loss = policy_loss + value_loss - entropy_coef * entropy_bonus
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_bonus": entropy_bonus,
            "total_loss": total_loss,
        }
