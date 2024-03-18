#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
linear dygraph model
"""

import paddle
import numpy as np


class LinearNet(paddle.nn.Layer):
    """model"""

    def __init__(self):
        """__init__"""
        paddle.set_default_dtype(np.float64)
        super(LinearNet, self).__init__()
        self.fc1 = paddle.nn.Linear(
            in_features=10,
            out_features=20,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
        )
        self.fc2 = paddle.nn.Linear(
            in_features=20,
            out_features=2,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
        )

    def forward(self, inputs):
        """forward"""
        output = self.fc1(inputs)
        output = self.fc2(output)
        return output
