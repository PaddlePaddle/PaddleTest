"""
test GFLHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import GFLHead, FCOSFeat
from ppdet.modeling.losses.gfocal_loss import QualityFocalLoss
from module.test_module import Test


class Config:
    """
    test GFLHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "GFLHead"
        fcosfeat = FCOSFeat(feat_in=8, feat_out=8)
        qualityfocalloss = QualityFocalLoss()
        self.net = GFLHead(
            conv_feat=fcosfeat,
            dgqp_module=None,
            num_classes=3,
            fpn_stride=[8, 16, 32],
            prior_prob=0.01,
            loss_class=qualityfocalloss,
            loss_dfl="DistributionFocalLoss",
            loss_bbox="GIoULoss",
            reg_max=16,
            feat_in_chan=8,
            nms=None,
            nms_pre=100,
            cell_offset=0,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1024, 4])
        label2 = paddle.rand(shape=[4, 256, 4])
        label3 = paddle.rand(shape=[4, 64, 4])
        self.label = [label1, label2, label3]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
