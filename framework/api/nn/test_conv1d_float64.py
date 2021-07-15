#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test Conv1D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import paddle.fluid as fluid
import pytest
import numpy as np


class TestConv1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        self.delta = 0.0001
        paddle.set_default_dtype("float64")
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestConv1d(paddle.nn.Conv1D)


@pytest.mark.api_nn_Conv1D_vartype
def test_conv1d_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    res = np.array([[[4.3166, 4.1242]], [[3.9617, 3.8575]]])
    obj.base(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    res = np.array([[[4.3166, 4.1242]], [[3.9617, 3.8575]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d1():
    """
    kernel_size = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = 2
    stride = 1
    padding = 0
    res = np.array([[[2.9210, 2.5172, 3.0026]], [[2.7743, 2.2806, 2.7643]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d2():
    """
    kernel_size = [2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 3
    kernel_size = [2]
    stride = 1
    padding = 0
    res = np.array(
        [
            [[2.9210, 2.5172, 3.0026], [2.9210, 2.5172, 3.0026], [2.9210, 2.5172, 3.0026]],
            [[2.7743, 2.2806, 2.7643], [2.7743, 2.2806, 2.7643], [2.7743, 2.2806, 2.7643]],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d3():
    """
    kernel_size = [3] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 2
    padding = 0
    res = np.array([[[4.3166]], [[3.9617]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d4():
    """
    kernel_size = [3] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 2
    padding = 1
    res = np.array([[[2.9210, 4.1242]], [[2.7743, 3.8575]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d5():
    """
    kernel_size = [3] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 6
    kernel_size = [3]
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[1.1094], [1.1094], [1.0751], [1.0751], [2.1321], [2.1321]],
            [[0.5946], [0.5946], [1.7757], [1.7757], [1.5914], [1.5914]],
        ]
    )
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        weight_attr=fluid.initializer.ConstantInitializer(value=1),
        bias_attr=fluid.initializer.ConstantInitializer(value=0),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d6():
    """
    kernel_size = [3] stride = 1 padding=0 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    stride = 1
    padding = 0
    res = np.array([[[2.7216, 2.5869], [2.7216, 2.5869]], [[2.4732, 2.4003], [2.4732, 2.4003]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d7():
    """
    kernel_size = [3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NLC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4]).transpose(0, 2, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3]
    stride = 1
    padding = 0
    data_format = "NLC"
    res = np.array([[[2.7216, 2.5869], [2.7216, 2.5869]], [[2.4732, 2.4003], [2.4732, 2.4003]]]).transpose(0, 2, 1)
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        data_format=data_format,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d8():
    """
    padding_mode = "reflect"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    padding_mode = "reflect"
    res = np.array([[[2.5298, 2.7216, 2.5869, 2.7787]], [[2.4073, 2.4732, 2.4003, 2.4662]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d9():
    """
    padding_mode = "replicate"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    padding_mode = "replicate"
    res = np.array([[[3.0042, 2.7216, 2.5869, 2.9267]], [[2.8188, 2.4732, 2.4003, 2.7388]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d10():
    """
    padding_mode = "circular"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    padding_mode = "circular"
    res = np.array([[[2.8695, 2.7216, 2.5869, 3.0614]], [[2.7458, 2.4732, 2.4003, 2.8117]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    dilation = 2
    padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d12():
    """
    padding_mode = "zeros" dilation = [2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = 1
    dilation = [2]
    padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = [1]
    padding = 1
    dilation = (2,)
    padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d14():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = 1
    padding = (1,)
    dilation = (2,)
    padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )


@pytest.mark.api_nn_Conv1D_parameters
def test_conv1d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3]
    stride = [1]
    padding = [1]
    dilation = (2,)
    padding_mode = "zeros"
    res = np.array([[[1.6100, 1.9365]], [[1.5691, 1.7079]]])
    obj.run(
        res=res,
        data=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
        bias_attr=fluid.initializer.ConstantInitializer(value=-0.3),
    )
