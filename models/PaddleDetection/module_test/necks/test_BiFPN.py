from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import BiFPN
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "BiFPN"
        self.net = BiFPN(in_channels=[8 ,16 ,32], out_channel=4, num_extra_levels=2, fpn_strides=[8, 16, 32, 64, 128], num_stacks=1, use_weighted_fusion=True, norm_type='bn', norm_groups=32, act='swish')
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 4, 32, 32])
        label2 = paddle.rand(shape=[4, 4, 16, 16])
        label3 = paddle.rand(shape=[4, 4, 8, 8])
        label4 = paddle.rand(shape=[4, 4, 4, 4])
        label5 = paddle.rand(shape=[4, 4, 2, 2])
        self.label = [label1, label2, label3, label4, label5]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
