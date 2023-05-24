from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import PicoHeadV2
from module.test_module import Test
from ppdet.modeling.losses.varifocal_loss import VarifocalLoss
from ppdet.modeling.heads import PicoFeat

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "PicoHeadV2"
        varifocalloss = VarifocalLoss()
        picofeat = PicoFeat(feat_in=8, feat_out=8)
        self.net = PicoHeadV2(conv_feat=picofeat, dgqp_module=None, num_classes=3, fpn_stride=[8 ,16 ,32], prior_prob=0.01, use_align_head=True, loss_class=varifocalloss, loss_dfl='DistributionFocalLoss', loss_bbox='GIoULoss', static_assigner_epoch=60, static_assigner='ATSSAssigner', assigner='TaskAlignedAssigner', reg_max=16, feat_in_chan=8, nms=None, nms_pre=100, act='hard_swish')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 4])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
