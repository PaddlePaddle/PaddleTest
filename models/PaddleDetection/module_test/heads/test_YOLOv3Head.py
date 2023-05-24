from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import YOLOv3Head
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "YOLOv3Head"
        self.net = YOLOv3Head(in_channels=[8 ,16 ,32], anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], num_classes=3, loss='YOLOv3Loss', iou_aware=False, iou_aware_factor=0.4, data_format='NCHW')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 24, 32, 32])
        label2 = paddle.rand(shape=[4, 24, 16, 16])
        label3 = paddle.rand(shape=[4, 24, 8, 8])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
