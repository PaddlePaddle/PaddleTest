#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.functional.conv2d_transpose
"""
from apibase import APIBase
from apibase import randtool
import paddle

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
def test_functional_conv2d_transpose_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.0)
    bias = np.full(shape=[out_channels], fill_value=0.0)
    stride = 1
    padding = 1
    output_padding = 0
    dilation = 1
    res = np.array([[[[5.9235, 5.9235], [5.9235, 5.9235]]], [[[5.5386, 5.5386], [5.5386, 5.5386]]]])

    obj.base(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    dilation = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.0)
    bias = np.full(shape=[out_channels], fill_value=0.0)
    res = np.array(
        [
            [
                [
                    [1.7994, 2.9210, 2.9210, 1.1216],
                    [3.1950, 5.9235, 5.9235, 2.7286],
                    [3.1950, 5.9235, 5.9235, 2.7286],
                    [1.3956, 3.0026, 3.0026, 1.6070],
                ]
            ],
            [
                [
                    [1.6811, 2.7743, 2.7743, 1.0932],
                    [2.8684, 5.5386, 5.5386, 2.6701],
                    [2.8684, 5.5386, 5.5386, 2.6701],
                    [1.1874, 2.7643, 2.7643, 1.5769],
                ]
            ],
        ]
    )
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose1():
    """
    w = 3.3 bias = -1.7
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=3.3)
    bias = np.full(shape=[out_channels], fill_value=-1.7)
    stride = 1
    padding = 1
    dilation = 1
    res = np.array([[[[17.8477, 17.8477], [17.8477, 17.8477]]], [[[16.5773, 16.5773], [16.5773, 16.5773]]]])
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose2():
    """
    padding = [1, 0]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1, 0]
    dilation = 1
    res = np.array(
        [
            [[[3.1950, 5.9235, 5.9235, 2.7286], [3.1950, 5.9235, 5.9235, 2.7286]]],
            [[[2.8684, 5.5386, 5.5386, 2.6701], [2.8684, 5.5386, 5.5386, 2.6701]]],
        ]
    )
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose3():
    """
    dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2, 2]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1, 0]
    dilation = 2
    # res.shape = [2, 1, 4, 6]
    res = np.array(
        [
            [
                [
                    [1.3956, 1.6070, 1.3956, 1.6070, 1.3956, 1.6070],
                    [1.7994, 1.1216, 1.7994, 1.1216, 1.7994, 1.1216],
                    [1.3956, 1.6070, 1.3956, 1.6070, 1.3956, 1.6070],
                    [1.7994, 1.1216, 1.7994, 1.1216, 1.7994, 1.1216],
                ]
            ],
            [
                [
                    [1.1874, 1.5769, 1.1874, 1.5769, 1.1874, 1.5769],
                    [1.6811, 1.0932, 1.6811, 1.0932, 1.6811, 1.0932],
                    [1.1874, 1.5769, 1.1874, 1.5769, 1.1874, 1.5769],
                    [1.6811, 1.0932, 1.6811, 1.0932, 1.6811, 1.0932],
                ]
            ],
        ]
    )
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose4():
    """
    out_channels = 3 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32")
    groups = 3
    in_channels = 6
    out_channels = 3
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1, 0]
    output_padding = 0
    dilation = 1
    # res.shape = [2, 3, 2, 4]
    res = np.array(
        [
            [
                [[1.5495, 3.3981, 3.3981, 1.8486], [1.5495, 3.3981, 3.3981, 1.8486]],
                [[1.8887, 4.1032, 4.1032, 2.2146], [1.8887, 4.1032, 4.1032, 2.2146]],
                [[2.6252, 3.9608, 3.9608, 1.3356], [2.6252, 3.9608, 3.9608, 1.3356]],
            ],
            [
                [[2.7245, 4.4060, 4.4060, 1.6815], [2.7245, 4.4060, 4.4060, 1.6815]],
                [[1.7327, 3.1496, 3.1496, 1.4169], [1.7327, 3.1496, 3.1496, 1.4169]],
                [[2.3186, 3.6639, 3.6639, 1.3453], [2.3186, 3.6639, 3.6639, 1.3453]],
            ],
        ]
    )
    obj.run(
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


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose5():
    """
    out_channels = 3 groups=3 data_format="NHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32").transpose(0, 2, 3, 1)
    in_channels = 6
    out_channels = 3
    groups = 3
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1, 0]
    output_padding = 0
    dilation = 1
    # res.shape = [2, 2, 4, 3], in "NHWC" format
    res = np.array(
        [
            [
                [[1.5495, 3.3981, 3.3981, 1.8486], [1.5495, 3.3981, 3.3981, 1.8486]],
                [[1.8887, 4.1032, 4.1032, 2.2146], [1.8887, 4.1032, 4.1032, 2.2146]],
                [[2.6252, 3.9608, 3.9608, 1.3356], [2.6252, 3.9608, 3.9608, 1.3356]],
            ],
            [
                [[2.7245, 4.4060, 4.4060, 1.6815], [2.7245, 4.4060, 4.4060, 1.6815]],
                [[1.7327, 3.1496, 3.1496, 1.4169], [1.7327, 3.1496, 3.1496, 1.4169]],
                [[2.3186, 3.6639, 3.6639, 1.3453], [2.3186, 3.6639, 3.6639, 1.3453]],
            ],
        ]
    ).transpose(0, 2, 3, 1)
    obj.run(
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
        data_format="NHWC",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose6():
    """
    out_channels = 3, groups=3, data_format="NHWC", output_padding=1,
    stride = 2, padding = [1, 0]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 6, 2, 2]).astype("float32").transpose(0, 2, 3, 1)
    in_channels = 6
    out_channels = 3
    groups = 3
    kernel_size = [3, 3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 2
    padding = [1, 0]
    dilation = 1
    output_padding = 1
    # res.shape = [2, 4, 6, 3] in "NHWC" format
    res = np.array(
        [
            [
                [
                    [1.1189, 1.1189, 1.7539, 0.6350, 0.6350, 0.0000],
                    [1.5495, 1.5495, 3.3981, 1.8486, 1.8486, 0.0000],
                    [0.4306, 0.4306, 1.6442, 1.2136, 1.2136, 0.0000],
                    [0.4306, 0.4306, 1.6442, 1.2136, 1.2136, 0.0000],
                ],
                [
                    [0.7600, 0.7600, 1.5980, 0.8380, 0.8380, 0.0000],
                    [1.8887, 1.8887, 4.1032, 2.2146, 2.2146, 0.0000],
                    [1.1287, 1.1287, 2.5052, 1.3766, 1.3766, 0.0000],
                    [1.1287, 1.1287, 2.5052, 1.3766, 1.3766, 0.0000],
                ],
                [
                    [1.6015, 1.6015, 2.3433, 0.7418, 0.7418, 0.0000],
                    [2.6252, 2.6252, 3.9608, 1.3356, 1.3356, 0.0000],
                    [1.0237, 1.0237, 1.6175, 0.5937, 0.5937, 0.0000],
                    [1.0237, 1.0237, 1.6175, 0.5937, 0.5937, 0.0000],
                ],
            ],
            [
                [
                    [0.8007, 0.8007, 1.6049, 0.8042, 0.8042, 0.0000],
                    [2.7245, 2.7245, 4.4060, 1.6815, 1.6815, 0.0000],
                    [1.9238, 1.9238, 2.8011, 0.8773, 0.8773, 0.0000],
                    [1.9238, 1.9238, 2.8011, 0.8773, 0.8773, 0.0000],
                ],
                [
                    [1.0835, 1.0835, 1.5807, 0.4973, 0.4973, 0.0000],
                    [1.7327, 1.7327, 3.1496, 1.4169, 1.4169, 0.0000],
                    [0.6492, 0.6492, 1.5689, 0.9196, 0.9196, 0.0000],
                    [0.6492, 0.6492, 1.5689, 0.9196, 0.9196, 0.0000],
                ],
                [
                    [0.8349, 0.8349, 1.9073, 1.0724, 1.0724, 0.0000],
                    [2.3186, 2.3186, 3.6639, 1.3453, 1.3453, 0.0000],
                    [1.4836, 1.4836, 1.7566, 0.2730, 0.2730, 0.0000],
                    [1.4836, 1.4836, 1.7566, 0.2730, 0.2730, 0.0000],
                ],
            ],
        ]
    ).transpose(0, 2, 3, 1)
    obj.run(
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
        data_format="NHWC",
    )
