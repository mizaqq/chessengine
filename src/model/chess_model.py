from torch import nn
import torch
from torch.distributions.utils import logits_to_probs
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x  # Save the input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Add the input back (the "shortcut")
        return F.relu(out)


class ChessPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(*[ResBlock(64) for _ in range(2)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  # 1x1 conv to reduce depth
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 4674),  # Map to action space
        )

    def forward(self, obs, mask=None):
        # obs shape: (Batch, 111, 8, 8)
        x = self.conv_input(obs)
        x = self.res_tower(x)
        logits = self.policy_head(x)

        if mask is not None:
            # Illegal move masking: Set logits of illegal moves to a very low value
            # mask shape: (Batch, 4674)
            logits = torch.where(
                mask.bool(), logits, torch.tensor(-1e9).to(logits.device)
            )

        return logits


class ChessPolicyProbs(nn.Module):
    def __init__(self, num_filters=128):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(20, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(*[ResBlock(num_filters) for _ in range(10)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),  # 1x1 conv to reduce depth
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4674),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),  # reduce to 1 channel
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, mask=None):
        x = self.conv_input(obs)
        x = self.res_tower(x)
        logits = self.policy_head(x)
        if mask is not None:
            logits = torch.where(
                mask.bool(), logits, torch.tensor(-1e9).to(logits.device)
            )
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(x)
        return probs, value
