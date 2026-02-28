import torch
from src.losses.composed_loss import ComposedLoss


def test_composed_loss_returns_named_terms_and_total():
    loss = ComposedLoss()
    out = loss.compute(
        policy_loss=torch.tensor(1.0),
        value_loss=torch.tensor(2.0),
        entropy_bonus=torch.tensor(0.5),
        entropy_coef=0.1,
    )
    assert set(out.keys()) == {"policy_loss", "value_loss", "entropy_bonus", "total_loss"}
