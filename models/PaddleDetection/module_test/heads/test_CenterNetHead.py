"""
test CenterNetHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import CenterNetHead
from module.test_module import Test


class Config:
    """
    test CenterNetHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "CenterNetHead"
        self.net = CenterNetHead(
            in_channels=16,
            num_classes=3,
            head_planes=16,
            prior_bias=-2.19,
            regress_ltrb=True,
            size_loss="L1",
            loss_weight={
                "heatmap": 1.0,
                "size": 0.1,
                "offset": 1.0,
                "iou": 0.0,
            },
            add_iou=False,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 16, 16, 16])
        self.data = feat1
        inputs = {}
        self.input = {"inputs": inputs}
        label1 = paddle.rand(shape=[4, 3, 16, 16])
        self.label = label1


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
