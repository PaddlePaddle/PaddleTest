"""
test CascadeTwoFCHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import CascadeTwoFCHead
from module.test_module import Test


class Config:
    """
    test CascadeTwoFCHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "CascadeTwoFCHead"
        self.net = CascadeTwoFCHead(in_channel=16, out_channel=32, resolution=2, num_cascade_stage=3)
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 16, 2, 2])
        # feat2 = paddle.rand(shape=[4, 16, 16, 16])
        # feat3 = paddle.rand(shape=[4, 16, 8, 8])
        self.data = feat1
        label1 = paddle.rand(shape=[4, 32])
        self.label = label1


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
