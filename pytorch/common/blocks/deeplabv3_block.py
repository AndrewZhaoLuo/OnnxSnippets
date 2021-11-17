import torch
import torch.nn.functional as F
from torch import nn

"""DeepLabV3-LIke Block"""

class DeepLabV3Block(nn.Module):

    def __init__(self, branch_off, channels, dilation=1):
        super().__init__()
        down_channels = channels // 4
        self.main_branch = nn.Sequential(
            nn.Conv2d(channels, down_channels, 1),
            nn.ReLU(),
            nn.Conv2d(
                down_channels, down_channels, 3, padding=dilation, dilation=dilation
            ),
            nn.ReLU(),
            nn.Conv2d(down_channels, channels, 1),
        )

        self.second_branch = nn.Conv2d(channels, channels, 1) if branch_off else None

    def forward(self, x):
        main_branch_results = self.main_branch(x)
        if self.second_branch is not None:
            x = self.second_branch(x)

        return x + main_branch_results
