"""
test SOLOv2Head
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import SOLOv2Head
from module.test_module import Test


class Config:
    """
    test SOLOv2Head
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "SOLOv2Head"
        self.net = SOLOv2Head(
            num_classes=3,
            in_channels=32,
            seg_feat_channels=32,
            stacked_convs=4,
            num_grids=[40, 36, 24, 16, 12],
            kernel_out_channels=32,
            dcn_v2_stages=[],
            segm_strides=[8, 8, 16, 32, 32],
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 32, 32, 32])
        feat2 = paddle.rand(shape=[4, 32, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        feat4 = paddle.rand(shape=[4, 32, 8, 8])
        feat5 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3, feat4, feat5]
        label1 = paddle.rand(shape=[4, 32, 40, 40])
        label2 = paddle.rand(shape=[4, 32, 36, 36])
        label3 = paddle.rand(shape=[4, 32, 24, 24])
        label4 = paddle.rand(shape=[4, 32, 16, 16])
        label5 = paddle.rand(shape=[4, 32, 12, 12])
        self.label = [label1, label2, label3, label4, label5]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
