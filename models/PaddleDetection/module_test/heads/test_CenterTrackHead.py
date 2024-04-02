"""
test CenterTrackHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import CenterTrackHead
from module.test_module import Test


class Config:
    """
    test CenterTrackHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "CenterTrackHead"
        self.net = CenterTrackHead(
            in_channels=16,
            num_classes=3,
            head_planes=16,
            task="tracking",
            loss_weight={"tracking": 1.0, "ltrb_amodal": 0.1},
            add_ltrb_amodal=True,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[1, 16, 8, 16])
        self.data = feat1
        inputs = {}
        bboxes = paddle.rand(shape=[10, 6])
        bbox_inds = paddle.randint(low=0, high=100, shape=[10])
        topk_ys = paddle.rand(shape=[10, 1])
        topk_xs = paddle.rand(shape=[10, 1])
        self.input = {
            "inputs": inputs,
            "bboxes": bboxes,
            "bbox_inds": bbox_inds,
            "topk_ys": topk_ys,
            "topk_xs": topk_xs,
        }
        label1 = paddle.rand(shape=[10, 6])
        self.label = label1


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
