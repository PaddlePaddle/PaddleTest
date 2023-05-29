"""
test MaskHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import MaskHead, MaskFeat
from ppdet.modeling.cls_utils import _get_class_default_kwargs
from ppdet.modeling.heads.roi_extractor import RoIAlign
from module.test_module import Test


class Config:
    """
    test MaskHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "MaskHead"
        maskfeat = MaskFeat(in_channel=8, out_channel=8)
        self.net = MaskHead(
            head=maskfeat,
            roi_extractor=_get_class_default_kwargs(RoIAlign),
            mask_assigner="MaskAssigner",
            num_classes=3,
            share_bbox_feat=False,
            export_onnx=False,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 3])
        label2 = paddle.rand(shape=[4, 1344, 4])
        label3 = paddle.rand(shape=[1344, 1])
        self.label = [label1, label2, label3]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
