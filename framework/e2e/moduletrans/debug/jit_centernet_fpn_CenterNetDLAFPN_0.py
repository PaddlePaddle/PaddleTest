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
    paddle.seed(33)
    np.random.seed(33)
    paddle.set_default_dtype("float64")
    input = [
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 8, 256, 256]).astype("float64")),
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 16, 128, 128]).astype("float64")),
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 32, 64, 64]).astype("float64")),
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 64, 32, 32]).astype("float64")),
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 128, 16, 16]).astype("float64")),
        paddle.to_tensor(randtool("float", -1, 1, shape=[4, 256, 8, 8]).astype("float64")),
    ]
    net = ppdet.modeling.necks.centernet_fpn.CenterNetDLAFPN(in_channels=[8, 16, 32, 64, 128, 256])
    if to_static:
        net = paddle.jit.to_static(net)

    # print("net parameters is: ", net.parameters())

    # opt = paddle.optimizer.SGD(learning_rate=0.00001, parameters=net.parameters())
    # dygraph train
    for epoch in range(2):
        loss = net(input)
        # loss = loss[3]
        # loss.backward()
        # opt.step()
        # opt.clear_grad()
        print("epoch {} is over~~".format(epoch))

    return loss


dy_out_final = train(False)
# st_out_final = train(True)
# # 结果打印
print("dy_out_final", dy_out_final)
# print("st_out_final", st_out_final)
# print(np.array_equal(dy_out_final.numpy(), st_out_final.numpy()))
#
# print("diff is: ", dy_out_final - st_out_final)
