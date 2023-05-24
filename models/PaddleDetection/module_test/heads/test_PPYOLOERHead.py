from __future__ import absolute_import
import paddle
import os
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from ppdet.modeling.heads import PPYOLOERHead
from module.test_module import Test
from ppdet.modeling.assigners.rotated_task_aligned_assigner import RotatedTaskAlignedAssigner
from ppdet.modeling.assigners.fcosr_assigner import FCOSRAssigner
class Config:
    def __init__(self):
        paddle.seed(33)
        self.module_name = "PPYOLOERHead"
        rotatedassigner = RotatedTaskAlignedAssigner()
        fcosrassigner = FCOSRAssigner()
        self.net = PPYOLOERHead(in_channels=[576, 288, 144], num_classes=15, act='swish', fpn_strides=[32, 16, 8], grid_cell_offset=0.5, angle_max=90, use_varifocal_loss=True, static_assigner_epoch=-1, trt=False, export_onnx=False, static_assigner=fcosrassigner, assigner=rotatedassigner, nms='MultiClassNMS', loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.05})
        self.net.eval()
        feat1 = paddle.rand(shape=[2, 576, 32, 32])
        feat2 = paddle.rand(shape=[2, 288, 64, 64])
        feat3 = paddle.rand(shape=[2, 144, 128, 128])
        self.data = [feat1, feat2, feat3]
        label1 = paddle.rand(shape=[2, 21504, 5])
        self.label = label1

cfg = Config()
test = Test(cfg)
test.forward_test()
test.backward_test()
