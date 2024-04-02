#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_conv2d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalConv2d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 0.005
        self.rtol = 0.005
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True
        # self.no_grad_var = ["weight", "bias"]


obj = TestFunctionalConv2d(paddle.nn.functional.conv2d)


@pytest.mark.api_nn_conv2d_vartype
def test_conv2d_base():
    """
    base
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array([[[[12.5633, 10.7230], [13.3298, 13.2501]]], [[[14.4928, 12.5433], [15.1694, 13.6606]]]])
    obj.base(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array([[[[12.5633, 10.7230], [13.3298, 13.2501]]], [[[14.4928, 12.5433], [15.1694, 13.6606]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d1():
    """
    kernel_size = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 2, 2]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[5.6780, 3.9777, 4.8305], [4.8986, 5.0738, 5.9837], [5.1103, 7.2506, 7.0629]]],
            [[[5.2631, 5.0306, 6.2066], [7.6067, 5.8608, 5.8187], [7.2613, 6.7396, 6.0788]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d2():
    """
    kernel_size = [2, 2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[3, 3, 2, 2]).astype("float32")
    bias = np.zeros(shape=[3]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [
                [[5.6780, 3.9777, 4.8305], [4.8986, 5.0738, 5.9837], [5.1103, 7.2506, 7.0629]],
                [[5.6780, 3.9777, 4.8305], [4.8986, 5.0738, 5.9837], [5.1103, 7.2506, 7.0629]],
                [[5.6780, 3.9777, 4.8305], [4.8986, 5.0738, 5.9837], [5.1103, 7.2506, 7.0629]],
            ],
            [
                [[5.2631, 5.0306, 6.2066], [7.6067, 5.8608, 5.8187], [7.2613, 6.7396, 6.0788]],
                [[5.2631, 5.0306, 6.2066], [7.6067, 5.8608, 5.8187], [7.2613, 6.7396, 6.0788]],
                [[5.2631, 5.0306, 6.2066], [7.6067, 5.8608, 5.8187], [7.2613, 6.7396, 6.0788]],
            ],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d3():
    """
    kernel_size = [2, 2] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 0
    res = np.array([[[[12.5633]]], [[[14.4928]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d4():
    """
    kernel_size = [2, 2] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 1
    res = np.array([[[[5.6780, 6.7046], [7.8561, 13.2501]]], [[[5.2631, 8.0467], [10.6586, 13.6606]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d5():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[6, 1, 3, 3]).astype("float32")
    bias = np.zeros(shape=[6]).astype("float32")
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[[4.3166]], [[4.3166]], [[4.6029]], [[4.6029]], [[3.6437]], [[3.6437]]],
            [[[3.6364]], [[3.6364]], [[5.5694]], [[5.5694]], [[5.2870]], [[5.2870]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d6():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[2, 3, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    res = np.array(
        [
            [[[8.4943, 7.2061], [9.0308, 8.9750]], [[8.4943, 7.2061], [9.0308, 8.9750]]],
            [[[9.8450, 8.4803], [10.3186, 9.2624]], [[9.8450, 8.4803], [10.3186, 9.2624]]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d7():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4]).transpose(0, 2, 3, 1)
    weight = np.ones(shape=[2, 3, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    data_format = "NHWC"
    res = np.array(
        [
            [[[8.4943, 7.2061], [9.0308, 8.9750]], [[8.4943, 7.2061], [9.0308, 8.9750]]],
            [[[9.8450, 8.4803], [10.3186, 9.2624]], [[9.8450, 8.4803], [10.3186, 9.2624]]],
        ]
    ).transpose(0, 2, 3, 1)
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, data_format=data_format)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 1]
    padding = 1
    dilation = 2
    # padding_mode = "zeros"
    res = np.array([[[[6.3869, 6.1435]]], [[[6.4830, 6.3120]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d12():
    """
    padding_mode = "zeros" dilation = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 1]
    padding = 1
    dilation = [2, 2]
    res = np.array([[[[6.3869, 6.1435]]], [[[6.4830, 6.3120]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 1]
    padding = 1
    dilation = (2, 2)
    res = np.array([[[[6.3869, 6.1435]]], [[[6.4830, 6.3120]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d14():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 1]
    padding = (1, 2)
    dilation = (2, 2)
    res = np.array([[[[6.1435, 6.3869, 6.1435, 6.3869]]], [[[6.3120, 6.4830, 6.3120, 6.4830]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv2d_parameters
def test_conv2d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    weight = np.ones(shape=[1, 3, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = [2, 1]
    padding = [1, 2]
    dilation = (2, 2)
    res = np.array([[[[6.1435, 6.3869, 6.1435, 6.3869]]], [[[6.3120, 6.4830, 6.3120, 6.4830]]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
