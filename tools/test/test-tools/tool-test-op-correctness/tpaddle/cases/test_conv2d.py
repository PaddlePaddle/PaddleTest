#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_conv2d.py
* @author zhengtianyu
* @date 2020-08-28 16:30:26
* @brief test_conv2d
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import paddle.fluid as fluid
import numpy as np


class TestConv2d(APIBase):
    """
    test
    """
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3 * 5
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestConv2d(paddle.nn.Conv2D)


@pytest.mark.p0
def test_conv2d_base():
    """
    base
    """
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[12.5633, 10.7230], [13.3298, 13.2501]]],
           [[[14.4928, 12.5433], [15.1694, 13.6606]]]])
    obj.base(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[12.5633, 10.7230], [13.3298, 13.2501]]],
           [[[14.4928, 12.5433], [15.1694, 13.6606]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d1():
    """
    kernel_size = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [2, 2]
    stride = 1
    padding = 0
    res = np.array([[[[5.6780, 3.9777, 4.8305],
          [4.8986, 5.0738, 5.9837],
          [5.1103, 7.2506, 7.0629]]],
        [[[5.2631, 5.0306, 6.2066],
          [7.6067, 5.8608, 5.8187],
          [7.2613, 6.7396, 6.0788]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d2():
    """
    kernel_size = [2, 2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 3
    kernel_size = [2, 2]
    stride = 1
    padding = 0
    res = np.array([[[[5.6780, 3.9777, 4.8305],
          [4.8986, 5.0738, 5.9837],
          [5.1103, 7.2506, 7.0629]],
         [[5.6780, 3.9777, 4.8305],
          [4.8986, 5.0738, 5.9837],
          [5.1103, 7.2506, 7.0629]],
         [[5.6780, 3.9777, 4.8305],
          [4.8986, 5.0738, 5.9837],
          [5.1103, 7.2506, 7.0629]]],
        [[[5.2631, 5.0306, 6.2066],
          [7.6067, 5.8608, 5.8187],
          [7.2613, 6.7396, 6.0788]],
         [[5.2631, 5.0306, 6.2066],
          [7.6067, 5.8608, 5.8187],
          [7.2613, 6.7396, 6.0788]],
         [[5.2631, 5.0306, 6.2066],
          [7.6067, 5.8608, 5.8187],
          [7.2613, 6.7396, 6.0788]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d3():
    """
    kernel_size = [2, 2] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 0
    res = np.array([[[[12.5633]]],
        [[[14.4928]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d4():
    """
    kernel_size = [2, 2] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = 2
    padding = 1
    res = np.array([[[[5.6780, 6.7046],
          [7.8561, 13.2501]]],
        [[[5.2631, 8.0467],
          [10.6586, 13.6606]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d5():
    """
    kernel_size = [2, 2] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 6
    kernel_size = [3, 3]
    stride = 2
    padding = 0
    groups = 3
    res = np.array([[[[4.3166]],
         [[4.3166]],
         [[4.6029]],
         [[4.6029]],
         [[3.6437]],
         [[3.6437]]],
        [[[3.6364]],
         [[3.6364]],
         [[5.5694]],
         [[5.5694]],
         [[5.2870]],
         [[5.2870]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d6():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[8.4943, 7.2061],
          [9.0308, 8.9750]],
         [[8.4943, 7.2061],
          [9.0308, 8.9750]]],
        [[[9.8450, 8.4803],
          [10.3186, 9.2624]],
         [[9.8450, 8.4803],
          [10.3186, 9.2624]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
             bias_attr=fluid.initializer.ConstantInitializer(value=-0.3))


def test_conv2d7():
    """
    kernel_size = [3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4]).transpose(0, 2, 3, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3]
    stride = 1
    padding = 0
    data_format="NHWC"
    res = np.array([[[[8.4943, 7.2061],
          [9.0308, 8.9750]],
         [[8.4943, 7.2061],
          [9.0308, 8.9750]]],
        [[[9.8450, 8.4803],
          [10.3186, 9.2624]],
         [[9.8450, 8.4803],
          [10.3186, 9.2624]]]]).transpose(0, 2, 3, 1)
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, data_format=data_format,
             weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
             bias_attr=fluid.initializer.ConstantInitializer(value=-0.3))


def test_conv2d8():
    """
    padding_mode = "reflect"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "reflect"
    res = np.array([[[[10.8965, 11.4650, 10.1772, 10.7457],
          [11.1693, 13.3298, 13.2501, 15.4106]]],
        [[[11.5616, 13.0999, 12.1508, 13.6891],
          [15.1975, 15.1694, 13.6606, 13.6325]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d9():
    """
    padding_mode = "replicate"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "replicate"
    res = np.array([[[[14.0708, 11.8796, 9.9366, 10.3045],
          [12.3989, 13.3298, 13.2501, 14.4000]]],
        [[[11.6388, 12.2609, 11.9894, 13.6088],
          [16.7781, 15.1694, 13.6606, 13.7325]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d10():
    """
    padding_mode = "circular"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    padding_mode = "circular"
    res = np.array([[[[13.4084, 12.6462, 12.4637, 13.7390],
          [12.3192, 13.3298, 13.2501, 14.4797]]],
        [[[13.2020, 12.9375, 13.1067, 14.2273],
          [15.2694, 15.1694, 13.6606, 15.2412]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    dilation = 2
    padding_mode = "zeros"
    res = np.array([[[[6.3869, 6.1435]]],
        [[[6.4830, 6.3120]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d12():
    """
    padding_mode = "zeros" dilation = [2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    dilation = [2, 2]
    padding_mode = "zeros"
    res = np.array([[[[6.3869, 6.1435]]],
        [[[6.4830, 6.3120]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = 1
    dilation = (2, 2)
    padding_mode = "zeros"
    res = np.array([[[[6.3869, 6.1435]]],
        [[[6.4830, 6.3120]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d14():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = (1, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3]
    stride = [2, 1]
    padding = (1, 2)
    dilation = (2, 2)
    padding_mode = "zeros"
    res = np.array([[[[6.1435, 6.3869, 6.1435, 6.3869]]],
        [[[6.3120, 6.4830, 6.3120, 6.4830]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv2d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
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
    res = np.array([[[[6.1435, 6.3869, 6.1435, 6.3869]]],
        [[[6.3120, 6.4830, 6.3120, 6.4830]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))
