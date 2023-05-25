"""
test ssdhead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from module.test_module import Test
from ppdet.modeling.heads import SSDHead
from ppdet.modeling.cls_utils import _get_class_default_kwargs
from ppdet.modeling.layers import AnchorGeneratorSSD


class Config:
    """
    test ssdhead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "SSDHead"
        self.net = SSDHead(
            num_classes=3,
            in_channels=(16, 32, 16, 8, 8, 8),
            anchor_generator=_get_class_default_kwargs(AnchorGeneratorSSD),
            kernel_size=3,
            padding=1,
            use_sepconv=False,
            conv_decay=0.0,
            loss="SSDLoss",
            use_extra_head=False,
        )
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
