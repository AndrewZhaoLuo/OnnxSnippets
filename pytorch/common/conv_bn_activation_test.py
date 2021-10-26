import itertools
import random
import unittest

import torch
from pytorch.common import common_test, conv_bn_activation
from torch import nn


class TestConvBNActivation(unittest.TestCase):
    in_channels = [3, 6, 9]
    out_channels = [6, 9, 12]
    kernel_size = [1, 3, 5]
    stride = [1, 2]
    padding = [0, 1, 2]
    dilation = [1, 2]
    groups = [1, 3]
    activation = [nn.ReLU()]
    bias = [True]

    def get_generic_conditions(self, max_sample_size=100):
        result = random.sample(
            list(
                itertools.product(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                    self.activation,
                    self.bias,
                )
            ),
            max_sample_size,
        )

        return [
            {
                "in_channels": t[0],
                "out_channels": t[1],
                "kernel_size": t[2],
                "stride": t[3],
                "padding": t[4],
                "dilation": t[5],
                "groups": t[6],
                "activation": t[7],
                "bias": t[8],
            }
            for t in result
        ]

    def helper(self, constructor, ndim):
        conditions = self.get_generic_conditions()
        for condition in conditions:
            with self.subTest(**condition):
                with torch.no_grad():
                    module = constructor(**condition)
                    common_test.verify_module(
                        module, input_shape=[1, condition["in_channels"]] + [12] * ndim
                    )

    def test_conv1d_bn_relu(self):
        self.helper(conv_bn_activation.Conv1DBnActivation, 1)

    def test_conv2d_bn_relu(self):
        self.helper(conv_bn_activation.Conv2DBnActivation, 2)

    def test_conv3d_bn_relu(self):
        self.helper(conv_bn_activation.Conv3DBnActivation, 3)

    def test_conv1dtranspose_bn_relu(self):
        self.helper(conv_bn_activation.Conv1DTransposeBnActivation, 1)

    def test_conv2dtranspose_bn_relu(self):
        self.helper(conv_bn_activation.Conv2DTransposeBnActivation, 2)

    def test_conv3dtranspose_bn_relu(self):
        self.helper(conv_bn_activation.Conv3DTransposeBnActivation, 3)
