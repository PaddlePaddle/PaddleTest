#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
model use conv.bias
"""
import math
import numpy as np
import paddle
import paddle.nn as nn


class ConvLayer(nn.Layer):
    """conv2d layer"""

    def __init__(
        self, ch_in, ch_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, data_format="NHWC", bias=False
    ):
        super(ConvLayer, self).__init__()
        bias_attr = False
        # fan_in = ch_in * kernel_size ** 2
        # bound = 1 / math.sqrt(fan_in)
        param_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform())
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            data_format=data_format,
        )

    def forward(self, inputs):
        """forward"""
        out = self.conv(inputs)
        return out


class NetSetConstantBias(nn.Layer):
    """model"""

    def __init__(self, in_channel, out_channel, data_format):
        super(NetSetConstantBias, self).__init__()
        self.heatmap = nn.Sequential(
            ConvLayer(in_channel, in_channel, kernel_size=3, padding=1, bias=True, data_format=data_format),
            nn.ReLU(),
            ConvLayer(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True, data_format=data_format),
        )
        with paddle.no_grad():
            self.heatmap[2].conv.bias[:] = -2.19

    def forward(self, inputs):
        """forward"""
        out = self.heatmap(inputs)
        return out


if __name__ == "__main__":
    inputs = np.random.random(size=[5, 10, 10, 3]).astype(np.float32)
    inputs = paddle.to_tensor(inputs)
    net = NetSetConstantBias(in_channel=3, out_channel=5, data_format="NHWC")
    out = net(inputs)
    print(out.shape)
