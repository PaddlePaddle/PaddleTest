#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_transpose_conv1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalTransposeConv1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        paddle.set_default_dtype("float64")
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalTransposeConv1d(paddle.nn.functional.conv1d_transpose)


@pytest.mark.api_nn_transpose_conv1d_vartype
def test_transpose_conv1d_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 1
    dilation = 1
    # padding_mode = "zeros"
    # groups=1
    res = np.array([[[2.4252, 2.4252]], [[3.4984, 3.4984]]])
    obj.base(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 2, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = 0
    dilation = 1
    # padding_mode = "zeros"
    # groups=1
    res = np.array(
        [
            [[1.5298, 2.4252, 2.4252, 0.8953], [1.5298, 2.4252, 2.4252, 0.8953]],
            [[1.6651, 3.4984, 3.4984, 1.8332], [1.6651, 3.4984, 3.4984, 1.8332]],
        ]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d1():
    """
    w = 3.3 bias = -1.7
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 1, 3]).astype("float32") * 3.3
    bias = np.ones(shape=[1]).astype("float32") * -1.7
    stride = 1
    padding = 1
    dilation = 1
    # groups=1
    res = np.array([[[6.3030, 6.3030]], [[9.8446, 9.8446]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d2():
    """
    stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = [1]
    dilation = 1
    groups = 1
    res = np.array([[[1.5298, 2.4252, 0.8953]], [[1.6651, 3.4984, 1.8332]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d3():
    """
    dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = [1]
    dilation = 2
    groups = 1
    res = np.array([[[0.8953, 1.5298, 0.8953, 1.5298]], [[1.8332, 1.6651, 1.8332, 1.6651]]])
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d4():
    """
    out_channels = 3 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32")
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = [1]
    dilation = 1
    groups = 3
    res = np.array(
        [[[0.6985, 0.6985], [0.6712, 0.6712], [1.0554, 1.0554]], [[0.9729, 0.9729], [1.1670, 1.1670], [1.3584, 1.3584]]]
    )
    obj.run(res=res, x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups, dilation=dilation)


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d5():
    """
    out_channels = 3 groups=3 data_format="NLC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32").transpose(0, 2, 1)
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 1
    padding = [1]
    dilation = 1
    groups = 3
    res = np.array(
        [[[0.6985, 0.6985], [0.6712, 0.6712], [1.0554, 1.0554]], [[0.9729, 0.9729], [1.1670, 1.1670], [1.3584, 1.3584]]]
    ).transpose(0, 2, 1)
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        data_format="NLC",
    )


@pytest.mark.api_nn_transpose_conv1d_parameters
def test_transpose_conv1d6():
    """
    out_channels = 3 groups=3 data_format="NLC" output_padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).astype("float32").transpose(0, 2, 1)
    weight = np.ones(shape=[3, 1, 3]).astype("float32")
    bias = np.zeros(shape=[1]).astype("float32")
    stride = 2
    padding = [1]
    dilation = 1
    groups = 3
    output_padding = 1
    res = np.array(
        [
            [[0.2485, 0.6985, 0.4500, 0.4500], [0.4109, 0.6712, 0.2603, 0.2603], [0.8704, 1.0554, 0.1850, 0.1850]],
            [[0.0197, 0.9729, 0.9533, 0.9533], [0.6805, 1.1670, 0.4866, 0.4866], [0.9650, 1.3584, 0.3934, 0.3934]],
        ]
    ).transpose(0, 2, 1)
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        data_format="NLC",
        output_padding=output_padding,
    )
