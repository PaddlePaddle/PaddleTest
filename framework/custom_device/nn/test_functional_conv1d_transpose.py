#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.functional.conv1d_transpose
"""
from apibase import APIBase
from apibase import randtool
import paddle

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
        self.debug = True
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


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose7():
    """
    padding can be of this form:
    padding = [padding_before_L, padding_after_L].
    in this case, padding = [4, 4],
    output shape = [2, 2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = [4, 4]
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    res = np.array([[[4.552747, 3.5600357], [4.552747, 3.5600357]], [[3.8121715, 3.4146838], [3.8121715, 3.4146838]]])
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
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose8():
    """
    if data_format = "NCL",
    padding can be of this form:
    padding = [[0, 0], [0, 0], [padding_before_L, padding_after_L]].
    if data_format = "NLC",
    padding can be of this form:
    padding = [[0, 0], [padding_before_L, padding_after_L], [0, 0]].

    in this case, data_format = "NCL",
      padding = [[0, 0], [0, 0], [3, 4]],
    output shape = [2, 2, 3]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    # same with padding=[3, 4]
    padding = [[0, 0], [0, 0], [3, 4]]
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    res = np.array(
        [
            [[4.312544, 4.552747, 3.5600357], [4.312544, 4.552747, 3.5600357]],
            [[2.937841, 3.8121715, 3.4146838], [2.937841, 3.8121715, 3.4146838]],
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
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose9():
    """
    if data_format = "NCL",
    padding can be of this form:
    padding = [[0, 0], [0, 0], [padding_before_L, padding_after_L]].
    if data_format = "NLC",
    padding can be of this form:
    padding = [[0, 0], [padding_before_L, padding_after_L], [0, 0]].

    in this case, data_format = "NLC",
    padding = [[0, 0], [3, 4], [0, 0]],
    output shape = [2, 3, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 8]).transpose((0, 2, 1))
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    # same with padding=[3, 4]
    padding = [[0, 0], [3, 4], [0, 0]]
    data_format = "NLC"
    dilation = 1
    output_padding = 0
    res = np.array(
        [
            [[4.312544, 4.552747, 3.5600357], [4.312544, 4.552747, 3.5600357]],
            [[2.937841, 3.8121715, 3.4146838], [2.937841, 3.8121715, 3.4146838]],
        ]
    ).transpose((0, 2, 1))
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
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose10():
    """
    padding can be one of "same" or "valid" (case insensitive),
    in this case, padding = "SaME",
    output shape = [2, 2, 5]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 5])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = "SaME"
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    res = np.array(
        [
            [
                [2.2616124, 3.7053628, 3.5989437, 4.256527, 2.8127768],
                [2.2616124, 3.7053628, 3.5989437, 4.256527, 2.8127768],
            ],
            [[3.8671303, 5.039712, 4.397395, 3.6909916, 2.51841], [3.8671303, 5.039712, 4.397395, 3.6909916, 2.51841]],
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
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_parameters
def test_functional_conv1d_transpose11():
    """
    padding can be one of "same" or "valid" (case insensitive),

    in this case, padding = "vALiD", which means no padding and
    is the same as padding=0,
    output shape = [2, 2, 7]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 5])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = "vALiD"
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    res = np.array(
        [
            [
                [1.3985769, 2.2616124, 3.7053628, 3.5989437, 4.256527, 2.8127768, 1.5206189],
                [1.3985769, 2.2616124, 3.7053628, 3.5989437, 4.256527, 2.8127768, 1.5206189],
            ],
            [
                [1.7852614, 3.8671303, 5.039712, 4.397395, 3.6909916, 2.51841, 1.3754654],
                [1.7852614, 3.8671303, 5.039712, 4.397395, 3.6909916, 2.51841, 1.3754654],
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
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_exception
def test_functional_conv1d_transpose12():
    """
     padding = "circular",

    ValueError: Unknown padding: 'CIRCULAR'. It can only be 'SAME' or 'VALID'.
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 5])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [3]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = "circular"
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    obj.exception(
        etype=ValueError,
        mode="python",
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format=data_format,
    )


@pytest.mark.api_nn_functional_conv1d_transpose_exception
def test_functional_conv1d_transpose13():
    """
    invalid "output_size": when stride > 1, there are multiple possible
    [L_out],
    Expected L_out must be in [infer_shape, infer_shape + stride),
    in this case L_out must be in [863, 870),
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 123])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [9]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 7
    padding = 0
    data_format = "NCL"
    dilation = 1
    output_padding = 0
    for output_size in [862, 861, 870, 871]:
        output_size = [output_size]
        obj.exception(
            etype=ValueError,
            mode="python",
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            output_size=output_size,
            data_format=data_format,
        )


@pytest.mark.api_nn_functional_conv1d_transpose_exception
def test_functional_conv1d_transpose14():
    """
    data_format = "NLC"

    ValueError:(InvalidArgument) The number of input channels should be equal to filter channels for Op(conv_transpose)
    . But received: the input's channels is [10], the shape of input is [2, 3, 1, 10], the filter's channels is [3], the
     shape of filter is [3, 2, 5, 1].
    The data_format is NLC.The error may come from wrong data_format setting.
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 10])
    in_channels = 3
    out_channels = 2
    groups = 1
    kernel_size = [5]
    weight = np.full(shape=[in_channels, out_channels // groups] + kernel_size, fill_value=1)
    bias = np.full(shape=[out_channels], fill_value=0)
    stride = 1
    padding = 0
    data_format = "NLC"
    dilation = 1
    output_padding = 0
    obj.exception(
        etype=ValueError,
        mode="python",
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
        output_size=None,
        data_format=data_format,
    )
