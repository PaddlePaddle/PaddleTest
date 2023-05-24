from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import PPYOLOEHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "PPYOLOEHead"
        self.net = PPYOLOEHead(in_channels=[8 ,16 ,32], num_classes=3, act='swish', fpn_strides=(32, 16, 8), grid_cell_scale=5.0, grid_cell_offset=0.5, reg_max=16, reg_range=None, static_assigner_epoch=4, use_varifocal_loss=True, static_assigner='ATSSAssigner', assigner='TaskAlignedAssigner', nms='MultiClassNMS')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 4])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
