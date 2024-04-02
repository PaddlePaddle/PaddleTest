"""
test TTFHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import TTFHead
from module.test_module import Test


class Config:
    """
    test TTFHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "TTFHead"
        self.net = TTFHead(
            in_channels=8,
            num_classes=3,
            hm_head_planes=128,
            wh_head_planes=64,
            hm_head_conv_num=2,
            wh_head_conv_num=2,
            hm_loss="CTFocalLoss",
            wh_loss="GIoULoss",
            wh_offset_base=16.0,
            down_ratio=4,
            dcn_head=False,
            lite_head=False,
            norm_type="bn",
            ags_module=False,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        self.data = feat1
        label1 = paddle.rand(shape=[4, 3, 32, 32])
        label2 = paddle.rand(shape=[4, 4, 32, 32])
        self.label = [label1, label2]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
