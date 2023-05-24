from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import YOLOXHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "YOLOXHead"
        self.net = YOLOXHead(in_channels=[8 ,16 ,32], num_classes=3, width_mult=1.0, depthwise=False, feat_channels=8, fpn_strides=(8, 16, 32), l1_epoch=285, act='silu', nms='MultiClassNMS')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 3])
        label2 = paddle.rand(shape=[4, 1344, 4])
        label3 = paddle.rand(shape=[1344, 1])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
