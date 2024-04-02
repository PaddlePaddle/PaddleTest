"""
test TOODHead
"""
from __future__ import absolute_import
import os
import sys
import paddle

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import TOODHead
from module.test_module import Test


class Config:
    """
    test TOODHead
    """

    def __init__(self):
        paddle.seed(33)
        self.module_name = "TOODHead"
        self.net = TOODHead(
            num_classes=3,
            feat_channels=8,
            stacked_convs=6,
            fpn_strides=(8, 16, 32),
            grid_cell_scale=8,
            grid_cell_offset=0.5,
            norm_type="gn",
            norm_groups=8,
            static_assigner_epoch=4,
            use_align_head=True,
            nms="MultiClassNMS",
            static_assigner="ATSSAssigner",
            assigner="TaskAlignedAssigner",
        )
        self.net.eval()
        feat1 = paddle.rand(shape=[4, 8, 32, 32])
        feat2 = paddle.rand(shape=[4, 8, 16, 16])
        feat3 = paddle.rand(shape=[4, 8, 8, 8])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[4, 1344, 3])
        label2 = paddle.rand(shape=[4, 1344, 4])
        label3 = paddle.rand(shape=[1344, 4])
        label4 = paddle.rand(shape=[1344, 1])
        self.label = [label1, label2, label3, label4]


cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
