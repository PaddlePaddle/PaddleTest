#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
Conv2D dygraph model
"""
import paddle
import numpy as np


class Conv2DNet(paddle.nn.Layer):
    """model"""

    def __init__(self, dtype=np.float64, in_channels=3, out_channels=10, data_format="NCHW"):
        """__init__"""
        paddle.set_default_dtype(dtype)
        super(Conv2DNet, self).__init__()
        self._conv1 = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            padding_mode="zeros",
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
            data_format=data_format,
        )
        self._bn1 = paddle.nn.BatchNorm2D(
            out_channels,
            momentum=0.9,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            data_format=data_format,
            name=None,
        )

    def forward(self, inputs):
        """forward"""
        output = self._conv1(inputs)
        output = self._bn1(output)
        return output
