"""
test RetinaHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import RetinaHead
from ppdet.modeling.heads.retina_head import RetinaFeat
from ppdet.modeling.proposal_generator.anchor_generator import RetinaAnchorGenerator
from ppdet.modeling.heads.fcos_head import FCOSFeat
from module.test_module import Test


class Config:
    """
    test RetinaHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "RetinaHead"
        retinaanchorgenerator = RetinaAnchorGenerator()
        fcosfeat = FCOSFeat(feat_in=8, feat_out=8, num_convs=4)
        self.net = RetinaHead(
            num_classes=3,
            conv_feat=fcosfeat,
            anchor_generator=retinaanchorgenerator,
            bbox_assigner="MaxIoUAssigner",
            loss_class="FocalLoss",
            loss_bbox="SmoothL1Loss",
            nms="MultiClassNMS",
            prior_prob=0.01,
            nms_pre=100,
            weights=[1.0, 1.0, 1.0, 1.0],
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 36, 32, 32])
        label2 = paddle.rand(shape=[4, 36, 16, 16])
        label3 = paddle.rand(shape=[4, 36, 8, 8])
        self.label = [label1, label2, label3]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
