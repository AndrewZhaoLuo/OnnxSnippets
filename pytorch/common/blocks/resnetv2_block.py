import torch
from pytorch.common import conv_bn_activation
from torch import nn

"""Resnetv2 Block without downsampling. E.g. very simple residual structure"""


class ResnetV2Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.preprocess = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv1 = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
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
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x_old = x
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + x_old
