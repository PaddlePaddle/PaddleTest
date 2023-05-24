from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import DeformableDETRHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "DeformableDETRHead"
        self.net = DeformableDETRHead(num_classes=3, hidden_dim=16, nhead=8, num_mlp_layers=3, loss='DETRLoss')
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 16, 16, 16])
        feat3 = paddle.rand(shape=[4, 32, 8, 8])
        self.data = [feat1, feat2, feat3]
        item1 = paddle.rand(shape=[6, 2, 10, 16])
        item2 = paddle.rand(shape=[2, 16, 16])
        item3 = paddle.rand(shape=[2, 10, 2])
        self.input = (item1, item2, item3)
        label1 = paddle.rand(shape=[2, 10, 3])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
