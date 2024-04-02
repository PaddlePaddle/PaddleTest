#!/bin/env python
# -*- coding=utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det
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


def train(to_static):
    """train"""
    paddle.seed(33)
    np.random.seed(33)
    paddle.set_default_dtype("float32")
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
    if to_static:
        net = paddle.jit.to_static(net)

    # print("net parameters is: ", net.parameters())

    opt = paddle.optimizer.SGD(learning_rate=0.000001, parameters=net.parameters())
    # dygraph train
    for epoch in range(1):
        loss = net(**input)
        loss = loss["loss"]
        loss.backward()
        opt.step()
        opt.clear_grad()

    return loss


dy_out_final = train(False)
st_out_final = train(True)
# 结果打印
print("dy_out_final", dy_out_final)
print("st_out_final", st_out_final)
print(np.array_equal(dy_out_final.numpy(), st_out_final.numpy()))

# print("diff is: ", dy_out_final - st_out_final)
