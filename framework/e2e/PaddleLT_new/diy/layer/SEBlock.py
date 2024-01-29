#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
SE Block
"""
import math
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, Linear
from paddle.nn.initializer import Uniform


class SEBlock(nn.Layer):
    """SEBlock"""

    def __init__(self, num_channels, lr_mult, reduction_ratio=4, name=None):
        """init"""
        super(SEBlock, self).__init__()
        self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        med_ch = num_channels // reduction_ratio
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(learning_rate=lr_mult, initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_channels,
            weight_attr=ParamAttr(learning_rate=lr_mult, initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )

    def forward(self, inputs):
        """forward"""
        pool = self.pool2d_gap(inputs)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = paddle.clip(x=excitation, min=0, max=1)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = paddle.multiply(inputs, excitation)
        return out
