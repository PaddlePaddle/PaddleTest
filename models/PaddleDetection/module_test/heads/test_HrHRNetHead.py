from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import HrHRNetHead
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "HrHRNetHead"
        self.net = HrHRNetHead(num_joints=10, loss='HrHRNetLoss', swahr=False, width=8)
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 10, 64, 64])
        label2 = paddle.rand(shape=[4, 10, 64, 64])
        self.label = [label1, label2]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
