#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.functional.conv2d_transpose
"""
from apibase import APIBase
from apibase import randtool
import paddle

import paddle.nn as nn
import pytest
import numpy as np


class TestFunctionalConv2dTranspose(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5
        self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalConv2dTranspose(paddle.nn.functional.conv2d_transpose)


@pytest.mark.api_nn_functional_conv2d_transpose_vartype
def test_conv2d_transpose_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    weight = paddle.create_parameter(
        shape=[in_channels, out_channels] + kernel_size,
        dtype="float32",
        default_initializer=nn.initializer.Constant(value=1.0),
    )
    bias = paddle.create_parameter(
        shape=[out_channels], dtype="float32", is_bias=True, default_initializer=nn.initializer.Constant(value=0.0)
    )

    stride = 1
    padding = 1
    output_padding = 0
    dilation = 1
    groups = 1
    res = np.array([[[[5.9235, 5.9235], [5.9235, 5.9235]]], [[[5.5386, 5.5386], [5.5386, 5.5386]]]])
    # result = paddle.nn.functional.conv2d_transpose(
    #     x=paddle.to_tensor(x),
    #     weight=weight,
    #     bias=bias,
    #     stride=stride,
    #     padding=padding,
    #     output_padding=output_padding,
    #     dilation=dilation,
    #     groups=groups,
    #     output_size=None,
    #     data_format="NCHW",
    # )
    # print(result, res, sep=3*'\n')

    x = paddle.to_tensor(x)
    # kwargs = dict(
    #     x=x,
    #     weight=weight,
    #     bias=bias,
    #     stride=stride,
    #     padding=padding,
    #     output_padding=output_padding,
    #     dilation=dilation,
    #     groups=groups,
    #     output_size=None,
    #     data_format="NCHW")
    # print(kwargs)
    # copy.deepcopy(kwargs)

    obj.base(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format="NCHW",
    )
