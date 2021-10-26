import itertools
import random
import unittest

import torch
from pytorch.common import common_test
from pytorch.common.blocks import resnetv1_block
from torch import nn


class TestResnetV1Block(unittest.TestCase):
    in_channels = [3, 6, 9]

    def get_generic_conditions(self):
        result = list(itertools.product(self.in_channels))

        return [{"in_channels": t[0]} for t in result]

    def helper(self, constructor):
        conditions = self.get_generic_conditions()
        for condition in conditions:
            for spatial_condition in [50, 100]:
                with self.subTest(**condition):
                    with torch.no_grad():
                        module = constructor(**condition)
                        common_test.verify_module(
                            module,
                            input_shape=[1, condition["in_channels"]]
                            + [spatial_condition] * 2,
                        )

    def test_resnetv1_block(self):
        self.helper(resnetv1_block.ResnetV1Block)
