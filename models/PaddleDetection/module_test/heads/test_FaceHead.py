from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import FaceHead
from module.test_module import Test
from ppdet.modeling.cls_utils import _get_class_default_kwargs
from ppdet.modeling.layers import AnchorGeneratorSSD

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "FaceHead"
        self.net = FaceHead(num_classes=3, in_channels=[8, 8, 8, 8, 8, 8], anchor_generator=_get_class_default_kwargs(AnchorGeneratorSSD), kernel_size=3, padding=1, conv_decay=0., loss='SSDLoss')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        feat4 = paddle.rand(shape=[4, 8, 8, 8])
        feat5 = paddle.rand(shape=[4, 8, 8, 8])
        feat6 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3, feat4, feat5, feat6]
        label1 = paddle.rand(shape=[4096, 4])
        label2 = paddle.rand(shape=[1536, 4])
        label3 = paddle.rand(shape=[384, 4])
        label4 = paddle.rand(shape=[384, 4])
        label5 = paddle.rand(shape=[256, 4])
        label6 = paddle.rand(shape=[256, 4])
        self.label = [label1, label2, label3, label4, label5, label6]
        self.input = paddle.rand(shape=[8, 3, 640, 640])

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
