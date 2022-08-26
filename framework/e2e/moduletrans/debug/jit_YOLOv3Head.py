#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
det
"""
import copy
import numpy as np
import paddle
import ppdet

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
    paddle.enable_static()
    paddle.disable_static()
    paddle.seed(33)
    np.random.seed(33)
    paddle.set_default_dtype("float32")
    input = paddle.load("YOLOv3Head_input.bl")
    net = ppdet.modeling.heads.yolo_head.YOLOv3Head(
        in_channels=[512, 256],
        anchors=[[11, 18], [34, 47], [51, 126], [115, 71], [120, 195], [254, 235]],
        anchor_masks=[[3, 4, 5], [0, 1, 2]],
        num_classes=80,
        loss=ppdet.modeling.losses.yolo_loss.YOLOv3Loss(),
        iou_aware=False,
        iou_aware_factor=0.4,
        data_format="NCHW",
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
