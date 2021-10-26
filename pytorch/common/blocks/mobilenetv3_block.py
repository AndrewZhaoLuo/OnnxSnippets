import torch
from pytorch.common import conv_bn_activation
from torch import nn

"""MobilenetV3 Block - Like a MobilenetV2 Block with Squeeze and Excite"""


"""Taken from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py"""


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MobilenetV3Block(nn.Module):
    EXPANSION_FACTOR = 6

    def __init__(self, in_channels):
        # We only use the stride = 1 block with expansion factor 6
        # These are the ones with residual connections which are interesting
        # Furthermore we use ReLU activation instead of ReLU 6
        # And always use ReLU 6
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
        self.squeeze_and_excite = SELayer(expansion_channel_size)
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
        x = self.squeeze_and_excite(x)
        x = self.pointwise_convolution_linear(x)
        return x + x_old
