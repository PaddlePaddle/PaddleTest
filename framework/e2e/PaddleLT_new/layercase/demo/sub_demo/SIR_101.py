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

    def forward(
        self,
        var_0,  # (shape: [145, 24, 112, 112], dtype: paddle.float32, stop_gradient: False)
    ):
        """
        forward
        """
        var_1 = paddle.nn.functional.pooling.max_pool2d(
            var_0, kernel_size=3, stride=2, padding=1, return_mask=False, ceil_mode=False, data_format="NCHW", name=None
        )
        return var_1


def create_paddle_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.rand(shape=[145, 24, 112, 112], dtype=paddle.float32),)
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.random(size=[145, 24, 112, 112]).astype("float32"),)
    return inputs
