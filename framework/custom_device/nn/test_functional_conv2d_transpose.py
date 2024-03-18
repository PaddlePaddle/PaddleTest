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
        self.types = [np.float32, np.float64]
        self.delta = 1e-3 * 5
        self.rtol = 1e-3
        self.debug = True
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


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose7():
    """
    padding = [1, 2, 3, 4] means
    [padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]
    output shape = [2, 1, 9, 5]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8, 8]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [5, 5]
    stride = 1
    padding = [1, 2, 3, 4]
    dilation = 1
    output_padding = 0
    groups = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.8)
    bias = np.full(shape=[out_channels], fill_value=-3.5)
    res = np.array(
        [
            [
                [
                    [18.276384, 23.175291, 24.224417, 21.342535, 21.766434],
                    [28.096786, 35.86816, 34.140198, 32.540596, 33.094135],
                    [39.51788, 50.176624, 48.544403, 49.104454, 48.310417],
                    [50.599766, 64.00076, 62.326912, 63.24839, 62.956383],
                    [52.484863, 65.54077, 62.386032, 64.89106, 63.96353],
                    [48.278343, 64.10091, 61.14633, 66.76668, 65.0205],
                    [49.528942, 65.52585, 64.942245, 68.26376, 65.59061],
                    [38.107845, 51.21739, 50.53804, 51.699905, 50.374325],
                    [27.025955, 37.393257, 36.755527, 37.55596, 35.728367],
                ]
            ],
            [
                [
                    [20.063206, 24.324394, 21.653057, 18.543024, 22.015406],
                    [31.901234, 39.631554, 37.349907, 33.978558, 36.46432],
                    [42.496323, 53.24839, 48.763603, 48.000576, 51.100777],
                    [53.673508, 66.84298, 63.42083, 63.351364, 65.23676],
                    [47.639957, 62.37429, 60.58043, 63.73015, 65.662636],
                    [47.542297, 60.242855, 61.405975, 67.9498, 70.54128],
                    [49.434097, 62.60486, 60.594994, 66.11465, 72.32123],
                    [38.839005, 48.98802, 49.181293, 52.092636, 57.684772],
                    [27.66183, 35.39344, 34.52407, 36.74185, 43.548786],
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
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format="NCHW",
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose8():
    """
     when data_format="NCHW", padding can be of this form:
      [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]];
     when data_format="NHWC", padding can be of this form:
     [[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；
    in this case:
          padding = [[0, 0], [0, 0], [1, 2], [3, 4]]
          data_format = 'NCHW'
    output shape = [2, 1, 9, 5]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8, 8]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [5, 5]
    stride = 1
    # same with padding = [1, 2, 3, 4]
    padding = [[0, 0], [0, 0], [1, 2], [3, 4]]
    data_format = "NCHW"
    dilation = 1
    output_padding = 0
    groups = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.8)
    bias = np.full(shape=[out_channels], fill_value=-3.5)
    res = np.array(
        [
            [
                [
                    [18.276384, 23.175291, 24.224417, 21.342535, 21.766434],
                    [28.096786, 35.86816, 34.140198, 32.540596, 33.094135],
                    [39.51788, 50.176624, 48.544403, 49.104454, 48.310417],
                    [50.599766, 64.00076, 62.326912, 63.24839, 62.956383],
                    [52.484863, 65.54077, 62.386032, 64.89106, 63.96353],
                    [48.278343, 64.10091, 61.14633, 66.76668, 65.0205],
                    [49.528942, 65.52585, 64.942245, 68.26376, 65.59061],
                    [38.107845, 51.21739, 50.53804, 51.699905, 50.374325],
                    [27.025955, 37.393257, 36.755527, 37.55596, 35.728367],
                ]
            ],
            [
                [
                    [20.063206, 24.324394, 21.653057, 18.543024, 22.015406],
                    [31.901234, 39.631554, 37.349907, 33.978558, 36.46432],
                    [42.496323, 53.24839, 48.763603, 48.000576, 51.100777],
                    [53.673508, 66.84298, 63.42083, 63.351364, 65.23676],
                    [47.639957, 62.37429, 60.58043, 63.73015, 65.662636],
                    [47.542297, 60.242855, 61.405975, 67.9498, 70.54128],
                    [49.434097, 62.60486, 60.594994, 66.11465, 72.32123],
                    [38.839005, 48.98802, 49.181293, 52.092636, 57.684772],
                    [27.66183, 35.39344, 34.52407, 36.74185, 43.548786],
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
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose9():
    """
     when data_format="NCHW", padding can be of this form:
      [[0,0], [0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right]];
     when data_format="NHWC", padding can be of this form:
     [[0,0], [padding_height_top, padding_height_bottom], [padding_width_left, padding_width_right], [0,0]]；
    in this case:
            padding = [[0, 0], [1, 2], [3, 4], [0, 0]]
            data_format = 'NHWC'
    output shape = [2, 9, 5, 1]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8, 8]).astype("float32").transpose((0, 2, 3, 1))
    in_channels = 3
    out_channels = 1
    kernel_size = [5, 5]
    stride = 1
    # same with padding = [1, 2, 3, 4]
    padding = [[0, 0], [1, 2], [3, 4], [0, 0]]
    data_format = "NHWC"
    dilation = 1
    output_padding = 0
    groups = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.8)
    bias = np.full(shape=[out_channels], fill_value=-3.5)
    res = np.array(
        [
            [
                [
                    [18.276384, 23.175291, 24.224417, 21.342535, 21.766434],
                    [28.096786, 35.86816, 34.140198, 32.540596, 33.094135],
                    [39.51788, 50.176624, 48.544403, 49.104454, 48.310417],
                    [50.599766, 64.00076, 62.326912, 63.24839, 62.956383],
                    [52.484863, 65.54077, 62.386032, 64.89106, 63.96353],
                    [48.278343, 64.10091, 61.14633, 66.76668, 65.0205],
                    [49.528942, 65.52585, 64.942245, 68.26376, 65.59061],
                    [38.107845, 51.21739, 50.53804, 51.699905, 50.374325],
                    [27.025955, 37.393257, 36.755527, 37.55596, 35.728367],
                ]
            ],
            [
                [
                    [20.063206, 24.324394, 21.653057, 18.543024, 22.015406],
                    [31.901234, 39.631554, 37.349907, 33.978558, 36.46432],
                    [42.496323, 53.24839, 48.763603, 48.000576, 51.100777],
                    [53.673508, 66.84298, 63.42083, 63.351364, 65.23676],
                    [47.639957, 62.37429, 60.58043, 63.73015, 65.662636],
                    [47.542297, 60.242855, 61.405975, 67.9498, 70.54128],
                    [49.434097, 62.60486, 60.594994, 66.11465, 72.32123],
                    [38.839005, 48.98802, 49.181293, 52.092636, 57.684772],
                    [27.66183, 35.39344, 34.52407, 36.74185, 43.548786],
                ]
            ],
        ]
    ).transpose((0, 2, 3, 1))
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
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose10():
    """
    padding can be one of "same" or "valid" (case insensitive),
    in this case, padding = "sAmE", so output and input will have the
    same height and width,

    output shape = [2, 1, 8, 8]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8, 8]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [5, 5]
    stride = 1
    padding = "sAmE"
    data_format = "NCHW"
    dilation = 1
    output_padding = 0
    groups = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.8)
    bias = np.full(shape=[out_channels], fill_value=-3.5)
    res = np.array(
        [
            [
                [
                    [21.938305, 28.096786, 35.86816, 34.140198, 32.540596, 33.094135, 26.935658, 19.164282],
                    [31.923145, 39.51788, 50.176624, 48.544403, 49.104454, 48.310417, 40.715683, 30.056938],
                    [39.645313, 50.599766, 64.00076, 62.326912, 63.24839, 62.956383, 52.001923, 38.60093],
                    [40.879837, 52.484863, 65.54077, 62.386032, 64.89106, 63.96353, 52.358498, 39.302593],
                    [36.06897, 48.278343, 64.10091, 61.14633, 66.76668, 65.0205, 52.81114, 36.98858],
                    [38.473793, 49.528942, 65.52585, 64.942245, 68.26376, 65.59061, 54.535458, 38.538548],
                    [28.488947, 38.107845, 51.21739, 50.53804, 51.699905, 50.374325, 40.755432, 27.645887],
                    [20.766783, 27.025955, 37.393257, 36.755527, 37.55596, 35.728367, 29.469196, 19.101898],
                ]
            ],
            [
                [
                    [23.23747, 31.901234, 39.631554, 37.349907, 33.978558, 36.46432, 27.800568, 20.070242],
                    [30.700386, 42.496323, 53.24839, 48.763603, 48.000576, 51.100777, 39.304836, 28.552773],
                    [39.057426, 53.673508, 66.84298, 63.42083, 63.351364, 65.23676, 50.62069, 37.451218],
                    [36.20568, 47.639957, 62.37429, 60.58043, 63.73015, 65.662636, 54.228367, 39.494022],
                    [35.572113, 47.542297, 60.242855, 61.405975, 67.9498, 70.54128, 58.57109, 45.87053],
                    [37.017036, 49.434097, 62.60486, 60.594994, 66.11465, 72.32123, 59.904167, 46.73341],
                    [29.554123, 38.839005, 48.98802, 49.181293, 52.092636, 57.684772, 48.399895, 38.250877],
                    [21.197083, 27.66183, 35.39344, 34.52407, 36.74185, 43.548786, 37.08404, 29.352432],
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
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv2d_transpose_parameters
def test_functional_conv2d_transpose11():
    """
    padding can be one of "same" or "valid" (case insensitive),
    in this case,  padding = "vaLiD", which means no padding and
    is the same as padding=0

    output shape = [2, 1, 6, 6]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4]).astype("float32")
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = "vaLiD"
    data_format = "NCHW"
    dilation = 1
    output_padding = 0
    groups = 1
    weight = np.full(shape=[in_channels, out_channels] + kernel_size, fill_value=1.8)
    bias = np.full(shape=[out_channels], fill_value=-3.5)
    res = np.array(
        [
            [
                [
                    [-5.1795936e-01, 1.7779684e00, 3.8765564e00, 2.3176346e00, 2.1706581e-02, -2.0768814e00],
                    [3.3469772e00, 6.7203894e00, 1.0506801e01, 8.5682983e00, 5.1948853e00, 1.4084749e00],
                    [5.5865173e00, 1.0595445e01, 1.9113880e01, 1.5801357e01, 1.0792427e01, 2.2739959e00],
                    [4.6771441e00, 1.0640911e01, 2.0493608e01, 2.0350107e01, 1.4386339e01, 4.5336428e00],
                    [8.1220770e-01, 5.6984901e00, 1.3863365e01, 1.4099443e01, 9.2131605e00, 1.0482864e00],
                    [-1.4273326e00, 1.8234339e00, 5.2562866e00, 6.8663855e00, 3.6156187e00, 1.8276572e-01],
                ]
            ],
            [
                [
                    [-1.5435842e00, -1.4150333e-01, 3.3530736e00, 3.5968418e00, 2.1947613e00, -1.2998157e00],
                    [2.6613412e00, 5.9735193e00, 1.1716458e01, 1.0984121e01, 7.6719437e00, 1.9290047e00],
                    [7.1410074e00, 1.3550493e01, 2.2587051e01, 1.9077953e01, 1.2668467e01, 3.6319065e00],
                    [7.5152550e00, 1.5685410e01, 2.3804956e01, 2.1089132e01, 1.2918980e01, 4.7994328e00],
                    [3.3103294e00, 9.5703869e00, 1.5441570e01, 1.3701853e01, 7.4417963e00, 1.5706124e00],
                    [-1.1693361e00, 1.9934125e00, 4.5709763e00, 5.6080227e00, 2.4452744e00, -1.3228941e-01],
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
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv2d_transpose_exception
def test_functional_conv2d_transpose12():
    """
    padding = "reflect"
     ValueError: Unknown padding: 'REFLECT'. It can only be 'SAME' or 'VALID'.
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4]).astype("float32")
    in_channels = 3
    out_channels = 1
    groups = 1
    kernel_size = [2, 2]
    stride = 1
    padding = "reflect"
    data_format = "NCHW"
    output_padding = 0
    dilation = 1
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1.0)
    bias = np.full(shape=[out_channels], fill_value=0.0)
    obj.exception(
        etype=ValueError,
        mode="python",
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv2d_transpose_exception
def test_functional_conv2d_transpose13():
    """
    invalid "output_size": when stride > 1, there are multiple possible
    [H_out, W_out],
    Expected output_size[i] must be in [infer_shape, infer_shape + strides[i]),
    in this case H_out must be in [863, 870),
    W_out must be in [611, 618)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [32, 58, 123, 87]).astype("float32")
    in_channels = 58
    out_channels = 21
    groups = 1
    kernel_size = [9, 9]
    stride = 7
    padding = 0
    output_padding = 0
    dilation = 1
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1.0)
    bias = np.full(shape=[out_channels], fill_value=0.0)
    for h_out, w_out in [(870, 611), (863, 618), (862, 619)]:
        output_size = [h_out, w_out]
        paddle.disable_static()
        obj.exception(
            etype=ValueError,
            mode="python",
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            output_size=output_size,
            data_format="NCHW",
        )


@pytest.mark.api_nn_functional_conv2d_transpose_exception
def test_functional_conv2d_transpose14():
    """
    (InvalidArgument) The number of input channels should be equal to filter channels for Op(conv_transpose).
    But received: the input's channels is [87], the shape of input is [32, 58, 123, 87], the filter's channels is [58],
    the shape of filter is [58, 21, 9, 9]. The data_format is NHWC.The error may come from wrong data_format setting.
      [Hint: Expected C == filter_dims[0], but received C:87 != filter_dims[0]:58.]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [32, 58, 123, 87]).astype("float32")
    in_channels = 58
    out_channels = 21
    groups = 1
    kernel_size = [9, 9]
    stride = 7
    padding = 0
    output_padding = 0
    dilation = 1
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1.0)
    bias = np.full(shape=[out_channels], fill_value=0.0)
    obj.exception(
        etype=ValueError,
        mode="python",
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        output_size=None,
        data_format="NHWC",
    )
