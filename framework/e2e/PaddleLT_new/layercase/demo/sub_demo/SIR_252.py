#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer
"""

import paddle
import numpy as np


class LayerCase(paddle.nn.Layer):
    """
    layercase
    """

    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[2304],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[768, 2304],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [6, 144, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        """
        forward
        """
        var_1 = paddle.nn.functional.norm.layer_norm(
            var_0, normalized_shape=[768], weight=self.parameter_0, bias=self.parameter_1, epsilon=1e-05
        )
        var_2 = paddle.nn.functional.common.linear(x=var_1, weight=self.parameter_3, bias=self.parameter_2, name=None)
        out = var_2.chunk(3, axis=-1)
        var_3 = out[0]
        var_4 = out[1]
        var_5 = out[2]
        return var_3, var_4, var_5


def create_paddle_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.rand(shape=[6, 144, 768], dtype=paddle.float32),)
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.random((6, 144, 768)).astype("float32"),)
    return inputs
