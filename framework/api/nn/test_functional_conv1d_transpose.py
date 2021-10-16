#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Conv1DTranspose
"""
from apibase import APIBase
from apibase import randtool
import paddle
import paddle.fluid as fluid
import pytest
import numpy as np


class TestTransposeConv1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3
        self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestTransposeConv1d(paddle.nn.Conv1DTranspose)


@pytest.mark.api_nn_Conv1DTranspose_vartype
def test_transpose_conv1d_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    dilation = 1
    # padding_mode = "zeros"
    # groups = 1
    res = np.array([[[2.4252, 2.4252]], [[3.4984, 3.4984]]])
    obj.base(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )
