from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import FCOSHead, FCOSFeat
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "FCOSHead"
        fcosfeat = FCOSFeat(feat_in=256, feat_out=256)
        self.net = FCOSHead(num_classes=3, fcos_feat=fcosfeat, fpn_stride=[8, 16, 32], prior_prob=0.01, multiply_strides_reg_targets=False, norm_reg_targets=True, centerness_on_reg=True, num_shift=0.5, sqrt_score=False, fcos_loss='FCOSLoss', nms='MultiClassNMS', trt=False)
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 256, 32, 32])
        feat2 = paddle.rand(shape=[4, 256, 16, 16])
        feat3 = paddle.rand(shape=[4, 256, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 3, 32, 32])
        label2 = paddle.rand(shape=[4, 3, 16, 16])
        label3 = paddle.rand(shape=[4, 3, 8, 8])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
