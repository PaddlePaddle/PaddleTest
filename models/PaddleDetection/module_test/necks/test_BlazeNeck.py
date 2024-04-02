"""
test BlazeNeck
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import BlazeNeck
from module.test_module import Test


class Config:
    """
    test BlazeNeck
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "BlazeNeck"
        self.net = BlazeNeck(in_channel=[96, 96], neck_type="fpn_ssh", data_format="NCHW")
        feat1 = paddle.rand(shape=[4, 96, 32, 32])
        feat2 = paddle.rand(shape=[4, 96, 16, 16])
        self.data = [feat1, feat2]
        label1 = paddle.rand(shape=[4, 48, 32, 32])
        label2 = paddle.rand(shape=[4, 48, 16, 16])
        self.label = [label1, label2]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
