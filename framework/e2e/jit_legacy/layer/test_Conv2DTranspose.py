#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Conv2D_Transpose
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


class Conv2DTransposeNet(paddle.nn.Layer):
    """
    a Conv2DTranpose Layer
    """

    def __init__(self, dtype=np.float32):
        paddle.set_default_dtype(dtype)
        super(Conv2DTransposeNet, self).__init__()
        self._conv2d_t = paddle.nn.Conv2DTranspose(in_channels=4, out_channels=8, kernel_size=3, padding=1)

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        return self._conv2d_t(inputs)


@pytest.mark.jit_Conv2D_Transpose_vartype
def test_jit_Conv2D_Transpose_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.Conv2D_Transpose(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "uint8"]
    """
    inps = randtool("float", -2, 2, shape=[2, 4, 224, 224])
    runner = Runner(func=Conv2DTransposeNet, name="Conv2D_Transpose_base", dtype=["float32", "float64"], ftype="layer")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
