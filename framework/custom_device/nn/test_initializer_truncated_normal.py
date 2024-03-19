#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# """
# test initializer_truncated_normal
# """
# from apibase import APIBase
# from apibase import randtool
# import pytest
# import paddle
# import paddle.fluid as fluid
# import numpy as np
#
#
# class TestInitializerTruncatedNormal(APIBase):
#    """
#    test
#    """
#
#    def hook(self):
#        """
#        implement
#        """
#        self.types = [np.float32]
#        self.delta = 1e-3 * 5
#
#
# obj = TestInitializerTruncatedNormal(paddle.nn.Conv2D)
# obj.places = [paddle.CUDAPlace(0)]
# obj.enable_backward = False
#
#
# @pytest.mark.api_initializer_truncated_normal_vartype
# def test_initializer_truncated_normal_base():
#    """
#    base
#    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0, name=None)
#    """
#    x = randtool("float", 0, 1, [2, 3, 4, 4])
#    in_channels = 3
#    out_channels = 1
#    kernel_size = [3, 3]
#    stride = 1
#    padding = 0
#    res = np.array(
#        [[[[4.4024887, 3.054892], [2.8605604, 8.828342]]], [[[6.5328727, 6.325475], [3.2522826, 4.8627253]]]]
#    )
#    if paddle.device.is_compiled_with_cuda() is True:
#        obj.base(
#            res=res,
#            data=x,
#            in_channels=in_channels,
#            out_channels=out_channels,
#            kernel_size=kernel_size,
#            stride=stride,
#            padding=padding,
#            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0, name=None),
#            bias_attr=False,
#        )
#
#
# @pytest.mark.api_initializer_truncated_normal_parameters
# def test_initializer_truncated_normal1():
#    """
#    kernel_size = [2, 2] stride = 2 padding=0 groups=3
#    weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=2.0)
#    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.4, std=0.8, name=None)
#    """
#    np.random.seed(obj.seed)
#    x = randtool("float", 0, 1, [2, 3, 4, 4])
#    in_channels = 3
#    out_channels = 6
#    kernel_size = [3, 3]
#    stride = 2
#    padding = 0
#    groups = 3
#    res = np.array(
#        [
#            [[[2.9801123]], [[3.6852546]], [[3.0435174]], [[1.6776688]], [[1.4755638]], [[1.1517941]]],
#            [[[2.869055]], [[2.445255]], [[3.3517156]], [[3.1544464]], [[2.1826687]], [[2.4802525]]],
#        ]
#    )
#    if paddle.device.is_compiled_with_cuda() is True:
#        obj.run(
#            res=res,
#            data=x,
#            in_channels=in_channels,
#            out_channels=out_channels,
#            kernel_size=kernel_size,
#            stride=stride,
#            padding=padding,
#            groups=groups,
#            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=0.4, std=0.8, name=None),
#            bias_attr=False,
#        )
#
#
# @pytest.mark.api_initializer_truncated_normal_parameters
# def test_initializer_truncated_normal2():
#    """
#    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NHWC"
#    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=1.1, std=0.2, name=None)
#    """
#    np.random.seed(obj.seed)
#    x = randtool("float", 0, 1, [2, 3, 4, 4]).transpose(0, 2, 3, 1)
#    in_channels = 3
#    out_channels = 2
#    kernel_size = [3, 3]
#    stride = 1
#    padding = 0
#    data_format = "NHWC"
#    res = np.array(
#        [
#            [[[14.70009, 13.828174], [12.406252, 11.928827]], [[15.234874, 14.700462], [16.340734, 14.942281]]],
#            [[[17.248663, 16.481497], [15.062735, 14.185494]], [[17.336819, 16.778688], [15.999238, 15.04023]]],
#        ]
#    )
#    if paddle.device.is_compiled_with_cuda() is True:
#        obj.run(
#            res=res,
#            data=x,
#            in_channels=in_channels,
#            out_channels=out_channels,
#            kernel_size=kernel_size,
#            stride=stride,
#            padding=padding,
#            data_format=data_format,
#            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=1.1, std=0.2, name=None),
#            bias_attr=False,
#        )
#
#
# @pytest.mark.api_initializer_truncated_normal_parameters
# def test_initializer_truncated_normal3():
#    """
#    padding_mode = "reflect"
#    weight_attr=paddle.nn.initializer.TruncatedNormal(mean=-2.1, std=0.1, name=None)
#    """
#    np.random.seed(obj.seed)
#    x = randtool("float", 0, 1, [2, 3, 4, 4])
#    in_channels = 3
#    out_channels = 1
#    kernel_size = [3, 3]
#    stride = [2, 1]
#    padding = 1
#    padding_mode = "reflect"
#    res = np.array(
#        [
#            [[[-22.479158, -23.755316, -20.885399, -22.21933], [-22.787048, -27.70649, -26.942293, -31.966446]]],
#            [[[-23.71504, -27.123177, -24.871805, -28.21984], [-31.329628, -31.530552, -28.201048, -28.149847]]],
#        ]
#    )
#    if paddle.device.is_compiled_with_cuda() is True:
#        obj.run(
#            res=res,
#            data=x,
#            in_channels=in_channels,
#            out_channels=out_channels,
#            kernel_size=kernel_size,
#            stride=stride,
#            padding=padding,
#            padding_mode=padding_mode,
#            weight_attr=paddle.nn.initializer.TruncatedNormal(mean=-2.1, std=0.1, name=None),
#            bias_attr=False,
#        )
