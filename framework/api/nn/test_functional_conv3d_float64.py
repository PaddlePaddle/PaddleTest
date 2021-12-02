#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_conv3d_float64
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalConv3d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        # self.no_grad_var = ["weight", "bias"]
        paddle.set_default_dtype("float64")
        self.delta = 0.005
        self.rtol = 0.005
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalConv3d(paddle.nn.functional.conv3d)


@pytest.mark.api_nn_conv3d_vartype
def test_conv3d_base():
    """
    base
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[[41.1337, 38.3068], [43.4962, 41.8764]], [[40.5446, 35.9235], [44.1236, 39.3930]]]],
            [[[[41.1783, 40.6048], [38.8515, 39.3217]], [[41.8694, 41.8558], [43.3938, 43.9873]]]],
        ]
    )
    obj.base(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[[41.1337, 38.3068], [43.4962, 41.8764]], [[40.5446, 35.9235], [44.1236, 39.3930]]]],
            [[[[41.1783, 40.6048], [38.8515, 39.3217]], [[41.8694, 41.8558], [43.3938, 43.9873]]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d1():
    """
    kernel_size = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 2, 2, 2]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [[10.2772, 10.1530, 10.9078], [12.0972, 12.3482, 12.6883], [13.0722, 14.6254, 14.2059]],
                    [[11.2427, 9.8658, 11.5452], [11.5212, 11.6108, 11.7400], [13.3817, 14.1561, 12.6881]],
                    [[11.5497, 9.8221, 12.0690], [13.6953, 11.9212, 10.5904], [13.2860, 12.9401, 10.7971]],
                ]
            ],
            [
                [
                    [[13.4966, 12.8346, 13.3530], [12.6845, 9.6804, 11.3862], [10.7447, 8.5911, 11.7397]],
                    [[12.2808, 13.5631, 13.3528], [11.8886, 11.2349, 11.0825], [11.7111, 11.3234, 12.9307]],
                    [[10.5355, 13.3375, 14.5260], [12.5133, 13.9420, 12.7570], [14.3575, 12.2793, 12.9360]],
                ]
            ],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d2():
    """
    kernel_size = [2, 2, 2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[3, 3, 2, 2, 2]).astype("float32")
    bias = np.zeros(shape=[3]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [
                    [[10.2772, 10.1530, 10.9078], [12.0972, 12.3482, 12.6883], [13.0722, 14.6254, 14.2059]],
                    [[11.2427, 9.8658, 11.5452], [11.5212, 11.6108, 11.7400], [13.3817, 14.1561, 12.6881]],
                    [[11.5497, 9.8221, 12.0690], [13.6953, 11.9212, 10.5904], [13.2860, 12.9401, 10.7971]],
                ],
                [
                    [[10.2772, 10.1530, 10.9078], [12.0972, 12.3482, 12.6883], [13.0722, 14.6254, 14.2059]],
                    [[11.2427, 9.8658, 11.5452], [11.5212, 11.6108, 11.7400], [13.3817, 14.1561, 12.6881]],
                    [[11.5497, 9.8221, 12.0690], [13.6953, 11.9212, 10.5904], [13.2860, 12.9401, 10.7971]],
                ],
                [
                    [[10.2772, 10.1530, 10.9078], [12.0972, 12.3482, 12.6883], [13.0722, 14.6254, 14.2059]],
                    [[11.2427, 9.8658, 11.5452], [11.5212, 11.6108, 11.7400], [13.3817, 14.1561, 12.6881]],
                    [[11.5497, 9.8221, 12.0690], [13.6953, 11.9212, 10.5904], [13.2860, 12.9401, 10.7971]],
                ],
            ],
            [
                [
                    [[13.4966, 12.8346, 13.3530], [12.6845, 9.6804, 11.3862], [10.7447, 8.5911, 11.7397]],
                    [[12.2808, 13.5631, 13.3528], [11.8886, 11.2349, 11.0825], [11.7111, 11.3234, 12.9307]],
                    [[10.5355, 13.3375, 14.5260], [12.5133, 13.9420, 12.7570], [14.3575, 12.2793, 12.9360]],
                ],
                [
                    [[13.4966, 12.8346, 13.3530], [12.6845, 9.6804, 11.3862], [10.7447, 8.5911, 11.7397]],
                    [[12.2808, 13.5631, 13.3528], [11.8886, 11.2349, 11.0825], [11.7111, 11.3234, 12.9307]],
                    [[10.5355, 13.3375, 14.5260], [12.5133, 13.9420, 12.7570], [14.3575, 12.2793, 12.9360]],
                ],
                [
                    [[13.4966, 12.8346, 13.3530], [12.6845, 9.6804, 11.3862], [10.7447, 8.5911, 11.7397]],
                    [[12.2808, 13.5631, 13.3528], [11.8886, 11.2349, 11.0825], [11.7111, 11.3234, 12.9307]],
                    [[10.5355, 13.3375, 14.5260], [12.5133, 13.9420, 12.7570], [14.3575, 12.2793, 12.9360]],
                ],
            ],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d3():
    """
    kernel_size = [3, 3, 3] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 0
    res = np.array([[[[[41.1337]]]], [[[[41.1783]]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d4():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 1
    res = np.array(
        [
            [[[[10.2772, 15.2571], [18.2521, 28.5663]], [[16.8695, 23.1947], [28.9226, 39.3930]]]],
            [[[[13.4966, 20.3464], [17.3102, 24.9590]], [[16.9992, 30.4095], [29.0104, 43.9873]]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d5():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[6, 1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[6]).astype("float32")
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[[[12.5633]]], [[[12.5633]]], [[[16.0763]]], [[[16.0763]]], [[[12.4942]]], [[[12.4942]]]],
            [[[[14.2857]]], [[[14.2857]]], [[[15.0920]]], [[[15.0920]]], [[[11.8007]]], [[[11.8007]]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d6():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[2, 3, 3, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [[[28.4936, 26.5148], [30.1473, 29.0135]], [[28.0812, 24.8464], [30.5865, 27.2751]]],
                [[[28.4936, 26.5148], [30.1473, 29.0135]], [[28.0812, 24.8464], [30.5865, 27.2751]]],
            ],
            [
                [[[28.5248, 28.1234], [26.8961, 27.2252]], [[29.0086, 28.9991], [30.0757, 30.4911]]],
                [[[28.5248, 28.1234], [26.8961, 27.2252]], [[29.0086, 28.9991], [30.0757, 30.4911]]],
            ],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d7():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NDHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4]).transpose(0, 2, 3, 4, 1)
    weight = np.ones(shape=[2, 3, 3, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    data_format = "NDHWC"
    res = np.array(
        [
            [
                [[[28.4936, 26.5148], [30.1473, 29.0135]], [[28.0812, 24.8464], [30.5865, 27.2751]]],
                [[[28.4936, 26.5148], [30.1473, 29.0135]], [[28.0812, 24.8464], [30.5865, 27.2751]]],
            ],
            [
                [[[28.5248, 28.1234], [26.8961, 27.2252]], [[29.0086, 28.9991], [30.0757, 30.4911]]],
                [[[28.5248, 28.1234], [26.8961, 27.2252]], [[29.0086, 28.9991], [30.0757, 30.4911]]],
            ],
        ]
    ).transpose(0, 2, 3, 4, 1)
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, data_format=data_format)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 2, 1]
    padding = 1
    dilation = 2
    # padding_mode = "zeros"
    res = np.array([[[[[11.3474, 13.2012]]]], [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d12():
    """
    padding_mode = "zeros" dilation = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 2, 1]
    padding = 1
    dilation = [2, 2, 2]
    res = np.array([[[[[11.3474, 13.2012]]]], [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 2, 1]
    padding = 1
    dilation = (2, 2, 2)
    res = np.array([[[[[11.3474, 13.2012]]]], [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d14():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 2, 1]
    padding = (1, 2, 2)
    dilation = (2, 2, 2)
    res = np.array(
        [
            [[[[14.4826, 8.5365, 14.4826, 8.5365], [14.4826, 8.5365, 14.4826, 8.5365]]]],
            [[[[14.1577, 11.3797, 14.1577, 11.3797], [14.1577, 11.3797, 14.1577, 11.3797]]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv3d_parameters
def test_conv3d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 2, 1]
    padding = [1, 2, 2]
    dilation = (2, 2, 2)
    res = np.array(
        [
            [[[[14.4826, 8.5365, 14.4826, 8.5365], [14.4826, 8.5365, 14.4826, 8.5365]]]],
            [[[[14.1577, 11.3797, 14.1577, 11.3797], [14.1577, 11.3797, 14.1577, 11.3797]]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
