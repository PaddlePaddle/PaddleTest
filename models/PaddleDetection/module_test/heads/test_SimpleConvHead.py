"""
test SimpleConvHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import SimpleConvHead
from module.test_module import Test


class Config:
    """
    test SimpleConvHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "SimpleConvHead"
        self.net = SimpleConvHead(
            num_classes=3,
            feat_in=32,
            feat_out=32,
            num_convs=1,
            fpn_strides=[32, 16, 8, 4],
            norm_type="gn",
            act="LeakyReLU",
            prior_prob=0.01,
            reg_max=16,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 32, 32, 32])
        feat2 = paddle.rand(shape=[4, 32, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        feat4 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3, feat4]
        label1 = paddle.rand(shape=[4, 1408, 68])
        self.label = label1


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
