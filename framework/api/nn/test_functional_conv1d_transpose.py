#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.functional.conv1d_transpose
"""
from apibase import APIBase
from apibase import randtool
import paddle

# import paddle.fluid as fluid
import pytest
import numpy as np


class TestFunctionalConv1dTranspose(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-3
        self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalConv1dTranspose(paddle.nn.functional.conv1d_transpose)


@pytest.mark.api_nn_functional_conv1d_transpose_vartype
def test_functional_conv1d_transpose_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    groups = 1
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = 1
    output_padding = 0
    dilation = 1
    # res.shape = [2, 1, 2]
    res = np.array([[[2.4252, 2.4252]], [[3.4984, 3.4984]]])
    obj.base(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    groups = 1
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1
    # res.shape = [2, 2, 4]
    res = np.array(
        [
            [[1.5298, 2.4252, 2.4252, 0.8953], [1.5298, 2.4252, 2.4252, 0.8953]],
            [[1.6651, 3.4984, 3.4984, 1.8332], [1.6651, 3.4984, 3.4984, 1.8332]],
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
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose1():
    """
    w = 3.3 bias = -1.7
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=3.3)
    bias = np.full(shape=[out_channels], fill_value=-1.7)
    stride = 1
    padding = 1
    output_padding = 0
    dilation = 1
    # res.shape = [2, 1, 2]
    res = np.array([[[6.3030, 6.3030]], [[9.8446, 9.8446]]])
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose2():
    """
    stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 2
    padding = [1]
    output_padding = 0
    dilation = 1
    # res.shape = [2, 1, 3]
    res = np.array([[[1.5298, 2.4252, 0.8953]], [[1.6651, 3.4984, 1.8332]]])
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose3():
    """
    dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 1
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1]
    output_padding = 0
    dilation = 2
    # res.shape = [2, 1, 4]
    res = np.array([[[0.8953, 1.5298, 0.8953, 1.5298]], [[1.8332, 1.6651, 1.8332, 1.6651]]])
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose4():
    """
    out_channels = 3 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2])
    in_channels = 3
    out_channels = 3
    groups = 3
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1]
    output_padding = 0
    dilation = 1
    # res.shape = [2, 3, 2]
    res = np.array(
        [[[0.6985, 0.6985], [0.6712, 0.6712], [1.0554, 1.0554]], [[0.9729, 0.9729], [1.1670, 1.1670], [1.3584, 1.3584]]]
    )
    obj.run(
        res=res,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NCL",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose5():
    """
    out_channels = 3 groups=3 data_format="NLC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).transpose(0, 2, 1)
    in_channels = 3
    out_channels = 3
    groups = 3
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [1]
    output_padding = 0
    dilation = 1
    # res.shape = [2, 2, 3] in "NLC" format
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
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NLC",
        name=None,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose6():
    """
    out_channels = 3, groups=3, data_format="NLC", output_padding=1,
    stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 2]).transpose(0, 2, 1)
    in_channels = 3
    out_channels = 3
    groups = 3
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 2
    padding = [1]
    dilation = 1
    output_padding = 1
    # res.shape = [2, 4, 3] in "NLC" format
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
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format="NLC",
        name=None,
    )
