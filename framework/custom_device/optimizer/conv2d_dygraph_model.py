#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
linear dygraph model
"""

import paddle
import numpy as np


class Conv2DNet(paddle.nn.Layer):
    """model"""

    def __init__(self):
        """__init__"""
        paddle.set_default_dtype(np.float64)
        super(Conv2DNet, self).__init__()
        self._conv1 = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode="zeros",
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
            data_format="NCHW",
        )
        self._bn1 = paddle.nn.BatchNorm2D(
            10, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format="NCHW", name=None
        )
        self._conv2 = paddle.nn.Conv2D(
            in_channels=10,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode="zeros",
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
            data_format="NCHW",
        )
        self._bn2 = paddle.nn.BatchNorm2D(
            2, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format="NCHW", name=None
        )

    def forward(self, inputs):
        """forward"""
        output = self._conv1(inputs)
        output = self._bn1(output)
        output = self._conv2(output)
        output = self._bn2(output)
        return output
