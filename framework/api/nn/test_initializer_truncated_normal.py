#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test initializer_truncated_normal
"""
from apibase import APIBase
from apibase import randtool
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np


class TestInitializerTruncatedNormal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5


obj = TestInitializerTruncatedNormal(paddle.nn.Conv2D)
obj.places = [paddle.CUDAPlace(0)]
obj.enable_backward = False


@pytest.mark.api_initializer_truncated_normal_vartype
def test_initializer_truncated_normal_base():
    """
    base
    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0, name=None)
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array(
        [[[[4.4024887, 3.054892], [2.8605604, 8.828342]]], [[[6.5328727, 6.325475], [3.2522826, 4.8627253]]]]
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
            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0, name=None),
            bias_attr=False,
        )


@pytest.mark.api_initializer_truncated_normal_parameters
def test_initializer_truncated_normal1():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.4, std=0.8, name=None)
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
            [[[2.7238693]], [[3.6058574]], [[2.8997347]], [[1.4733795]], [[1.3498653]], [[1.032864]]],
            [[[2.5999534]], [[2.3688996]], [[3.1804018]], [[2.898926]], [[2.0069914]], [[2.322328]]],
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
            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.4, std=0.8, name=None),
            bias_attr=False,
        )


@pytest.mark.api_initializer_truncated_normal_parameters
def test_initializer_truncated_normal2():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NHWC"
    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=1.1, std=0.2, name=None)
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
            [[[14.789145, 13.895285], [12.418382, 11.9859085]], [[15.121986, 14.735731], [16.481052, 15.031883]]],
            [[[17.364357, 16.616774], [15.155772, 14.271544]], [[17.29946, 16.850342], [16.018963, 15.10177]]],
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
            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=1.1, std=0.2, name=None),
            bias_attr=False,
        )


@pytest.mark.api_initializer_truncated_normal_parameters
def test_initializer_truncated_normal3():
    """
    padding_mode = "reflect"
    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=-2.1, std=0.1, name=None)
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
            [[[-21.011927, -22.0851, -19.5799, -20.778374], [-21.402576, -25.808033, -25.355343, -29.83345]]],
            [[[-22.185707, -25.252943, -23.313328, -26.41671], [-29.27957, -29.325695, -26.339834, -26.31158]]],
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
            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=-2.1, std=0.1, name=None),
            bias_attr=False,
        )
