import torch
from pytorch.common import conv_bn_activation
from torch import nn

"""MobilenetV2 Block with inverted residual structure"""


class MobilenetV2Block(nn.Module):
    EXPANSION_FACTOR = 6

    def __init__(self, in_channels):
        # We only use the stride = 1 block with expansion factor 6
        # These are the ones with residual connections which are interesting
        # Furthermore we use ReLU activation instead of ReLU 6
        super().__init__()

        expansion_channel_size = in_channels * self.EXPANSION_FACTOR

        self.pointwise_convolution = conv_bn_activation.Conv2DBnActivation(
            in_channels,
            expansion_channel_size,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.depthwise_convolution = conv_bn_activation.Conv2DBnActivation(
            expansion_channel_size,
            expansion_channel_size,
            kernel_size=3,
            padding=1,
            stride=1,
            dilation=1,
            groups=expansion_channel_size,
        )
        self.pointwise_convolution_linear = conv_bn_activation.Conv2DBnActivation(
            expansion_channel_size,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            activation=None,
        )

    def forward(self, x):
        x_old = x
        x = self.pointwise_convolution(x)
        x = self.depthwise_convolution(x)
        x = self.pointwise_convolution_linear(x)
        return x + x_old
