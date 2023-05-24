from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.necks import DilatedEncoder
from module.test_module import Test

class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "DilatedEncoder" 
        self.net = DilatedEncoder(in_channels=[16], out_channels=[8], block_mid_channels=128, num_residual_blocks=4, block_dilations=[2, 4, 6, 8])
        feat1 = paddle.rand(shape=[4, 16, 16, 16])
        self.data = [feat1]
        label1 = paddle.rand(shape=[4, 8, 16, 16])
        self.label = [label1]

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
