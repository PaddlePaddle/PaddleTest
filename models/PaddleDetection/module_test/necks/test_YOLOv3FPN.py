from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import YOLOv3FPN
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "YOLOv3FPN"
        self.net = YOLOv3FPN(in_channels=[8 ,16 ,32], norm_type='bn', freeze_norm=False, data_format='NCHW')
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1024, 8, 8])
        label2 = paddle.rand(shape=[4, 512, 16, 16])
        label3 = paddle.rand(shape=[4, 256, 32, 32])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
