from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import SparseRCNNHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "SparseRCNNHead"
        self.net = SparseRCNNHead(head_hidden_dim=256, head_dim_feedforward=2048, nhead=8, head_dropout=0.0, head_cls=1, head_reg=3, head_dim_dynamic=64, head_num_dynamic=2, head_num_heads=6, deep_supervision=True, num_proposals=100, num_classes=80)
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 256, 272, 328])
        feat2 = paddle.rand(shape=[4, 256, 136, 164])
        feat3 = paddle.rand(shape=[4, 256, 68, 82])
        feat4 = paddle.rand(shape=[4, 256, 34, 41])
        feat5 = paddle.rand(shape=[4, 256, 17, 21])
        self.data = [feat1, feat2, feat3, feat4, feat5]
        self.input = paddle.randint(100, shape=[4,4])
        label1 = paddle.rand(shape=[4, 1344, 3])
        label2 = paddle.rand(shape=[4, 1344, 4])
        label3 = paddle.rand(shape=[1344, 1])
        self.label = [label1, label2, label3]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
