from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import S2ANetHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "S2ANetHead"
        self.net = S2ANetHead(stacked_convs=2, feat_in=8, feat_out=8, num_classes=15, anchor_strides=[8, 16, 32, 64, 128], anchor_scales=[4], anchor_ratios=[1.0], target_means=0.0, target_stds=1.0, align_conv_type='AlignConv', align_conv_size=3, use_sigmoid_cls=True, reg_loss_weight=[1.0, 1.0, 1.0, 1.0, 1.1], cls_loss_weight=[1.1, 1.05], reg_loss_type='l1', nms_pre=2000, nms='MultiClassNMS')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1024, 15])
        label2 = paddle.rand(shape=[4, 256, 15])
        label3 = paddle.rand(shape=[4, 64, 15])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
