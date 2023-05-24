from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import YOLOFHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "YOLOFHead"
        self.net = YOLOFHead(num_classes=3, conv_feat='YOLOFFeat', anchor_generator='AnchorGenerator', bbox_assigner='UniformAssigner', loss_class='FocalLoss', loss_bbox='GIoULoss', ctr_clip=32.0, delta_mean=[0.0, 0.0, 0.0, 0.0], delta_std=[1.0, 1.0, 1.0, 1.0], nms='MultiClassNMS', prior_prob=0.01, nms_pre=1000, use_inside_anchor=False, trt=False, exclude_nms=False)
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
