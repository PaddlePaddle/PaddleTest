"""
test RoIAlignHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import RoIAlign
from module.test_module import Test


class Config:
    """
    test RoiAlignHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "RoIAlign"
        self.net = RoIAlign(
            resolution=8,
            spatial_scale=0.0625,
            sampling_ratio=0,
            canconical_level=4,
            canonical_size=16,
            start_level=0,
            end_level=0,
            aligned=False,
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[8, 8, 32, 32])
        feat2 = paddle.rand(shape=[8, 8, 16, 16])
        feat3 = paddle.rand(shape=[8, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        roi1 = paddle.rand(shape=[8, 8])
        roi2 = paddle.rand(shape=[8, 8])
        roi = [roi1, roi2]
        rois_num = paddle.randint(low=0, high=10, dtype="int32", shape=[4, 2])
        self.input = {"roi": roi, "rois_num": rois_num}
        label1 = paddle.rand(shape=[42, 8, 8, 8])
        self.label = label1


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
