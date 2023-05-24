from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import PPYOLOTinyFPN
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "PPYOLOTinyFPN"
        self.net = PPYOLOTinyFPN(in_channels=[80 ,56 ,34], detection_block_channels=[160, 128, 96], norm_type='bn', data_format='NCHW')
        feat1 = paddle.rand(shape=[4, 80, 32, 32])
        feat2 = paddle.rand(shape=[4, 56, 16, 16])
        feat3 = paddle.rand(shape=[4, 34, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 160, 8, 8])
        label2 = paddle.rand(shape=[4, 128, 16, 16])
        label3 = paddle.rand(shape=[4, 96, 32, 32])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
