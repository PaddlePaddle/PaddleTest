from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import CenterNetDLAFPN
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "CenterNetDLAFPN" 
        self.net = CenterNetDLAFPN(in_channels=[16, 32, 64, 128, 256, 512], down_ratio=4, last_level=3, out_channel=0, first_level=0, dcn_v2=False, with_sge=False)
        feat1 = paddle.rand(shape=[4, 16, 256, 256])
        feat2 = paddle.rand(shape=[4, 32, 128, 128])
        feat3 = paddle.rand(shape=[4, 64, 64, 64])
        feat4 = paddle.rand(shape=[4, 128, 32, 32])
        feat5 = paddle.rand(shape=[4, 256, 16, 16])
        feat6 = paddle.rand(shape=[4, 512, 8, 8])
        self.data = [feat1, feat2, feat3, feat4, feat5, feat6]
        label1 = paddle.rand(shape=[4, 16, 256, 256])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
