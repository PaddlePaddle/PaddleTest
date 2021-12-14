#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test jit sequential sublayer
"""
import os
import shutil
import paddle
import numpy as np
from tools import compare


pwd = os.getcwd()
save_path = os.path.join(pwd, "save_path")
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(os.path.join(pwd, "save_path"))


class BufferLayers(paddle.nn.Layer):
    """
    BufferLayers
    """

    def __init__(self, out_channel):
        super(BufferLayers, self).__init__()
        self.out_channel = out_channel

    def forward(self, x):
        """
        forward
        """
        mean = paddle.mean(x)
        if mean < 0.0:
            x = x * self._mask()

        out = x - mean
        return out

    def _mask(self):
        return paddle.to_tensor(np.zeros([self.out_channel], "float32"))


class SequentialNet(paddle.nn.Layer):
    """
    SequentialNet
    """

    def __init__(self, sub_layer, in_channel, out_channel):
        super(SequentialNet, self).__init__()
        self.layer = paddle.nn.Sequential(
            ("l1", paddle.nn.Linear(in_channel, in_channel)),
            ("l2", paddle.nn.Linear(in_channel, out_channel)),
            ("l3", sub_layer(out_channel)),
        )

    def forward(self, x):
        """
        forward
        """
        out = self.layer(x)
        return out


def test_jit_sequential_sublayer1():
    """
    class BufferLayers(paddle.nn.Layer)
    class SequentialNet(paddle.nn.Layer)
    test: net = SequentialNet(BufferLayers, 10, 3)
    the results after jit.to_static(net), jit.save/load(net)
    should be equal to origin
    """
    paddle.seed(1314)
    np.random.seed(1314)
    net_dy = SequentialNet(BufferLayers, 10, 3)
    net_st = paddle.jit.to_static(net_dy)

    x = paddle.rand([16, 10], "float32")
    out1 = net_dy(x)
    out2 = net_st(x)
    compare(out1.numpy(), out2.numpy())

    paddle.jit.save(net_st, os.path.join(save_path, "sequential_net"))
    load_net = paddle.jit.load(os.path.join(save_path, "sequential_net"))
    out3 = load_net(x)
    compare(out2.numpy(), out3.numpy())
