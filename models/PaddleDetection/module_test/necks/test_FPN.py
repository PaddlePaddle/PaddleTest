"""
test FPN
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import FPN
from module.test_module import Test


class Config:
    """
    test FPN
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "FPN"
        self.net = FPN(
            in_channels=[8, 16, 32],
            out_channel=8,
            norm_type=None,
            norm_decay=0.0,
            freeze_norm=False,
            relu_before_extra_convs=True,
        )
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 8, 32, 32])
        label2 = paddle.rand(shape=[4, 8, 16, 16])
        label3 = paddle.rand(shape=[4, 8, 8, 8])
        label4 = paddle.rand(shape=[4, 8, 4, 4])
        self.label = [label1, label2, label3, label4]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
