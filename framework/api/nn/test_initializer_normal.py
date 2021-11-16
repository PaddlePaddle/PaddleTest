#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_normal
"""
from apibase import APIBase
from apibase import randtool
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np


class TestInitializerNormal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5


obj = TestInitializerNormal(paddle.nn.Conv2D)
obj.places = [paddle.CUDAPlace(0)]
obj.enable_backward = False


@pytest.mark.api_initializer_normal_vartype
def test_initializer_normal_base():
    """
    base
    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array(
        [[[[9.696032, 6.2314734], [4.5925426, 19.060331]]], [[[14.223241, 13.581673], [6.131522, 9.923124]]]]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.base(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal1():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 6
    kernel_size = [3, 3]
    stride = 2
    padding = 0
    groups = 3
    res = np.array(
        [
            [[[3.4906466]], [[5.3631434]], [[3.3413303]], [[-0.3206752]], [[0.09324306]], [[-0.63597184]]],
            [[[3.7898493]], [[2.6968503]], [[2.8813796]], [[3.0387473]], [[0.18275034]], [[1.4571017]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal2():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NHWC"
    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4]).transpose(0, 2, 3, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    data_format = "NHWC"
    res = np.array(
        [
            [[[9.696032, 0.757347], [6.2314734, 1.9067024]], [[4.5925426, 0.7299702], [19.060331, 4.5685883]]],
            [[[14.223241, 6.7473884], [13.581673, 4.739414]], [[6.131522, 1.6402493], [9.923124, 0.7511649]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            data_format=data_format,
            weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal3():
    """
    padding_mode = "reflect"
    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "reflect"
    res = np.array(
        [
            [[[8.600252, 6.062214, 10.376897, 7.4796634], [14.477899, 4.5925436, 19.060333, 8.73203]]],
            [[[11.757825, 7.498006, 13.49066, 11.478115], [12.05721, 6.1315236, 9.923124, 10.59241]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal4():
    """
    padding_mode = "replicate"
    weight_attr=paddle.nn.initializer.Normal(mean=-1.0, std=2.0)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "replicate"
    res = np.array(
        [
            [[[-0.25635287, 0.6774132, -3.2480803, -7.2378592], [-3.132905, -8.737241, 5.810273, -1.7762027]]],
            [[[0.8862386, -0.61035603, 4.994542, -5.26088], [-6.272961, -9.037898, -3.7375047, -1.8376815]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.Normal(mean=-1.0, std=2.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal5():
    """
    padding_mode = "circular"
    weight_attr=paddle.nn.initializer.Normal(mean=2.0, std=5.0)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "circular"
    res = np.array(
        [
            [[[48.49079, 63.295036, 57.694126, 54.563183], [64.20126, 38.140923, 74.15096, 54.82368]]],
            [[[55.2387, 49.607075, 66.03619, 59.64029], [69.127945, 45.66765, 52.129078, 62.14041]]],
        ]
    )
    if paddle.device.is_compiled_with_cuda() is True:
        obj.run(
            res=res,
            data=x,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            weight_attr=paddle.nn.initializer.Normal(mean=2.0, std=5.0),
            bias_attr=False,
        )


@pytest.mark.api_initializer_normal_parameters
def test_initializer_normal6():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    weight_attr=paddle.nn.initializer.Normal(mean=-0.2, std=1.99)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = [1, 2]
    dilation = (2, 2)
    padding_mode = "zeros"
    res = np.array([[[[6.9537334, 3.4081156, 2.5187492, 6.7148337]]], [[[3.6040933, 3.8371656, 2.5087252, 6.355131]]]])
    if paddle.device.is_compiled_with_cuda() is True:
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
            weight_attr=paddle.nn.initializer.Normal(mean=-0.2, std=1.99),
            bias_attr=False,
        )
