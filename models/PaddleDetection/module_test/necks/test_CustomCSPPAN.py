"""
test CustomCSPPAN
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import CustomCSPPAN
from module.test_module import Test


class Config:
    """
    test CustomCSPPAN
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "CustomCSPPAN"
        self.net = CustomCSPPAN(
            in_channels=[8, 16, 32],
            out_channels=[32, 16, 8],
            norm_type="bn",
            act="swish",
            stage_num=1,
            block_num=3,
            spp=True,
        )
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 32, 8, 8])
        label2 = paddle.rand(shape=[4, 16, 16, 16])
        label3 = paddle.rand(shape=[4, 8, 32, 32])
        self.label = [label1, label2, label3]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
