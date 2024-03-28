#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_conv1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalConv1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalConv1d(paddle.nn.functional.conv1d)


@pytest.mark.api_nn_conv1d_vartype
def test_conv1d_base():
    """
    base
    """
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array([[[4.3166, 4.1242]], [[3.9617, 3.8575]]])
    obj.base(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array([[[4.3166, 4.1242]], [[3.9617, 3.8575]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d1():
    """
    kernel_size = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 2]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    res = np.array([[[2.9210, 2.5172, 3.0026]], [[2.7743, 2.2806, 2.7643]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d2():
    """
    kernel_size = [2, 2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[3, 3, 2]).astype("float32")
    bias = np.zeros(shape=[3]).astype("float32")
    stride = 1
    padding = 0
    res = np.array(
        [
            [[2.9210, 2.5172, 3.0026], [2.9210, 2.5172, 3.0026], [2.9210, 2.5172, 3.0026]],
            [[2.7743, 2.2806, 2.7643], [2.7743, 2.2806, 2.7643], [2.7743, 2.2806, 2.7643]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d3():
    """
    kernel_size = [2, 2] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 0
    res = np.array([[[4.3166]], [[3.9617]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d4():
    """
    kernel_size = [2, 2] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = 1
    res = np.array([[[2.9210, 4.1242]], [[2.7743, 3.8575]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d5():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[6, 1, 3]).astype("float32")
    bias = np.zeros(shape=[6]).astype("float32")
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[1.1094], [1.1094], [1.0751], [1.0751], [2.1321], [2.1321]],
            [[0.5946], [0.5946], [1.7757], [1.7757], [1.5914], [1.5914]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d6():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[2, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    res = np.array([[[2.7216, 2.5869], [2.7216, 2.5869]], [[2.4732, 2.4003], [2.4732, 2.4003]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d7():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NLC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4]).transpose(0, 2, 1)
    weight = np.ones(shape=[2, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[2]).astype("float32") * -0.3
    stride = 1
    padding = 0
    data_format = "NLC"
    res = np.array([[[2.7216, 2.5869], [2.7216, 2.5869]], [[2.4732, 2.4003], [2.4732, 2.4003]]]).transpose(0, 2, 1)
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, data_format=data_format)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[1]).astype("float32") * -0.3
    stride = [1]
    padding = 1
    dilation = 2
    # padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d12():
    """
    padding_mode = "zeros" dilation = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[1]).astype("float32") * -0.3
    stride = [1]
    padding = [1]
    dilation = (2,)
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[1]).astype("float32") * -0.3
    stride = [1]
    padding = [1]
    dilation = (2,)
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d14():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[1]).astype("float32") * -0.3
    stride = [1]
    padding = [1]
    dilation = (2,)
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_conv1d_parameters
def test_conv1d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    weight = np.ones(shape=[1, 3, 3]).astype("float32") * 0.7
    bias = np.ones(shape=[1]).astype("float32") * -0.3
    stride = [1]
    padding = [1]
    dilation = (2,)
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
