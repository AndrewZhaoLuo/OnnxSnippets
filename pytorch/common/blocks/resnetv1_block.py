import torch
from pytorch.common import conv_bn_activation
from torch import nn

"""Resnetv1 Block without downsampling. E.g. very simple residual structure"""


class ResnetV1Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.conv2 = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            activation=None,
        )
        self.final_activation = nn.ReLU()

    def forward(self, x):
        x_old = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.final_activation(x + x_old)
