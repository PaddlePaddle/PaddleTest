from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import DETRHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "DETRHead"
        self.net = DETRHead(num_classes=3, hidden_dim=16, nhead=8, num_mlp_layers=3, loss='DETRLoss', fpn_dims=[32, 8, 16], with_mask_head=False, use_focal_loss=False)
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        item1 = paddle.rand(shape=[6, 2, 10, 16])
        item2 = paddle.rand(shape=[2, 16, 6, 8])
        item3 = paddle.rand(shape=[2, 16, 6, 8])
        item4 = paddle.rand(shape=[2, 1, 1, 6, 8])
        self.input = (item1, item2, item3, item4)
        label1 = paddle.rand(shape=[2, 10, 4])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
