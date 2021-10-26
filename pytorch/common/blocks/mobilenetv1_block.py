import torch
from pytorch.common import conv_bn_activation
from torch import nn

"""MobilenetV1 Block consisting of a Depthwise Conv + Pointwise Conv."""


class MobilenetV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise_conv = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
