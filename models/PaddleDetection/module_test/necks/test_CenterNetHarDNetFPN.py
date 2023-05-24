from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import CenterNetHarDNetFPN
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "CenterNetHarDNetFPN" 
        self.net = CenterNetHarDNetFPN(in_channels=[96, 214, 458, 784], num_layers=85, down_ratio=4, first_level=None, last_level=4, out_channel=0)
        feat1 = paddle.rand(shape=[4, 96, 64, 64])
        feat2 = paddle.rand(shape=[4, 214, 32, 32])
        feat3 = paddle.rand(shape=[4, 458, 16, 16])
        feat4 = paddle.rand(shape=[4, 784, 8, 8])
        self.data = [feat1, feat2, feat3, feat4]
        label1 = paddle.rand(shape=[4, 200, 32, 32])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
