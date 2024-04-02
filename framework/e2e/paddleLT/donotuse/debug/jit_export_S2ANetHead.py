#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ocr rec_srn_head
"""
import copy
import numpy as np
import paddle
import ppdet
from ppdet.modeling.proposal_generator.target_layer import RBoxAssigner

paddle.seed(33)
np.random.seed(33)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


input = paddle.load("S2ANetHead_input.bl")
net = ppdet.modeling.heads.s2anet_head.S2ANetHead(
    stacked_convs=2,
    feat_in=256,
    feat_out=256,
    num_classes=9,
    anchor_strides=[8, 16, 32, 64, 128],
    anchor_scales=[4],
    anchor_ratios=[1.0],
    target_means=0.0,
    target_stds=1.0,
    align_conv_type="AlignConv",
    align_conv_size=3,
    use_sigmoid_cls=True,
    anchor_assign=RBoxAssigner(),
    reg_loss_weight=[1.0, 1.0, 1.0, 1.0, 1.05],
    cls_loss_weight=[1.05, 1.0],
    reg_loss_type="l1",
    nms_pre=2000,
    nms=ppdet.modeling.layers.MultiClassNMS(),
)
net = paddle.jit.to_static(net)
# print("net is: ", net.parameters())
res = net(**input)
#
paddle.jit.save(net, path="SERes5Head")
print("export success~~")
