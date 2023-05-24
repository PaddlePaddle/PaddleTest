from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import CascadeHead, CascadeXConvNormHead
from module.test_module import Test
from ppdet.modeling.cls_utils import _get_class_default_kwargs
from ppdet.modeling.heads import RoIAlign

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "CascadeHead"
        cascadexconvnormhead = CascadeXConvNormHead(in_channel=32, num_convs=4, conv_dim=32, out_channel=32, resolution=2, norm_type='gn', freeze_norm=False, num_cascade_stage=3) 
        self.net = CascadeHead(head=cascadexconvnormhead, in_channel=32, roi_extractor=_get_class_default_kwargs(RoIAlign), bbox_assigner='BboxAssigner', num_classes=3, bbox_weight=[[10., 10., 5., 5.], [20.0, 20.0, 10.0, 10.0], [30.0, 30.0, 15.0, 15.0]], num_cascade_stages=3, bbox_loss=None, reg_class_agnostic=True, stage_loss_weights=None, loss_normalize_pos=False, add_gt_as_proposals=[True, False, False])
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 32, 32, 32])
        feat2 = paddle.rand(shape=[4, 32, 16, 16])
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
