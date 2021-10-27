import itertools
import random
import unittest

import torch
from pytorch.common import common_test
from pytorch.common.blocks import resnext_block
from torch import nn


class TestResnetNeXtBlock(unittest.TestCase):
    in_channels = [16, 32]

    def get_generic_conditions(self):
        result = list(itertools.product(self.in_channels))

        return [{"in_channels": t[0], "out_channels": t[0]} for t in result]

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

    def test_resneXt_block(self):
        self.helper(resnext_block.ResNeXtBlock)
