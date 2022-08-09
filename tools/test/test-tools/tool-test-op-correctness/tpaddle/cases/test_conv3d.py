#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_conv3d.py
* @author zhengtianyu
* @date 2020-09-02 15:26:10
* @brief test_conv3d
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import paddle.fluid as fluid
import numpy as np


class TestConv3d(APIBase):
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


obj = TestConv3d(paddle.nn.Conv3D)


@pytest.mark.p0
def test_conv3d_base():
    """
    base
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[[41.1337, 38.3068],
           [43.4962, 41.8764]],
          [[40.5446, 35.9235],
           [44.1236, 39.3930]]]],
        [[[[41.1783, 40.6048],
           [38.8515, 39.3217]],
          [[41.8694, 41.8558],
           [43.3938, 43.9873]]]]])
    obj.base(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d():
    """
    default
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[[41.1337, 38.3068],
           [43.4962, 41.8764]],
          [[40.5446, 35.9235],
           [44.1236, 39.3930]]]],
        [[[[41.1783, 40.6048],
           [38.8515, 39.3217]],
          [[41.8694, 41.8558],
           [43.3938, 43.9873]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d1():
    """
    kernel_size = [2, 2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [2, 2, 2]
    stride = 1
    padding = 0
    res = np.array([[[[[10.2772, 10.1530, 10.9078],
           [12.0972, 12.3482, 12.6883],
           [13.0722, 14.6254, 14.2059]],
          [[11.2427, 9.8658, 11.5452],
           [11.5212, 11.6108, 11.7400],
           [13.3817, 14.1561, 12.6881]],
          [[11.5497, 9.8221, 12.0690],
           [13.6953, 11.9212, 10.5904],
           [13.2860, 12.9401, 10.7971]]]],
        [[[[13.4966, 12.8346, 13.3530],
           [12.6845, 9.6804, 11.3862],
           [10.7447, 8.5911, 11.7397]],
          [[12.2808, 13.5631, 13.3528],
           [11.8886, 11.2349, 11.0825],
           [11.7111, 11.3234, 12.9307]],
          [[10.5355, 13.3375, 14.5260],
           [12.5133, 13.9420, 12.7570],
           [14.3575, 12.2793, 12.9360]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d2():
    """
    kernel_size = [2, 2, 2], out_channels = 3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 3
    kernel_size = [2, 2, 2]
    stride = 1
    padding = 0
    res = np.array([[[[[10.2772, 10.1530, 10.9078],
           [12.0972, 12.3482, 12.6883],
           [13.0722, 14.6254, 14.2059]],
          [[11.2427, 9.8658, 11.5452],
           [11.5212, 11.6108, 11.7400],
           [13.3817, 14.1561, 12.6881]],
          [[11.5497, 9.8221, 12.0690],
           [13.6953, 11.9212, 10.5904],
           [13.2860, 12.9401, 10.7971]]],
         [[[10.2772, 10.1530, 10.9078],
           [12.0972, 12.3482, 12.6883],
           [13.0722, 14.6254, 14.2059]],
          [[11.2427, 9.8658, 11.5452],
           [11.5212, 11.6108, 11.7400],
           [13.3817, 14.1561, 12.6881]],
          [[11.5497, 9.8221, 12.0690],
           [13.6953, 11.9212, 10.5904],
           [13.2860, 12.9401, 10.7971]]],
         [[[10.2772, 10.1530, 10.9078],
           [12.0972, 12.3482, 12.6883],
           [13.0722, 14.6254, 14.2059]],
          [[11.2427, 9.8658, 11.5452],
           [11.5212, 11.6108, 11.7400],
           [13.3817, 14.1561, 12.6881]],
          [[11.5497, 9.8221, 12.0690],
           [13.6953, 11.9212, 10.5904],
           [13.2860, 12.9401, 10.7971]]]],
        [[[[13.4966, 12.8346, 13.3530],
           [12.6845, 9.6804, 11.3862],
           [10.7447, 8.5911, 11.7397]],
          [[12.2808, 13.5631, 13.3528],
           [11.8886, 11.2349, 11.0825],
           [11.7111, 11.3234, 12.9307]],
          [[10.5355, 13.3375, 14.5260],
           [12.5133, 13.9420, 12.7570],
           [14.3575, 12.2793, 12.9360]]],
         [[[13.4966, 12.8346, 13.3530],
           [12.6845, 9.6804, 11.3862],
           [10.7447, 8.5911, 11.7397]],
          [[12.2808, 13.5631, 13.3528],
           [11.8886, 11.2349, 11.0825],
           [11.7111, 11.3234, 12.9307]],
          [[10.5355, 13.3375, 14.5260],
           [12.5133, 13.9420, 12.7570],
           [14.3575, 12.2793, 12.9360]]],
         [[[13.4966, 12.8346, 13.3530],
           [12.6845, 9.6804, 11.3862],
           [10.7447, 8.5911, 11.7397]],
          [[12.2808, 13.5631, 13.3528],
           [11.8886, 11.2349, 11.0825],
           [11.7111, 11.3234, 12.9307]],
          [[10.5355, 13.3375, 14.5260],
           [12.5133, 13.9420, 12.7570],
           [14.3575, 12.2793, 12.9360]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d3():
    """
    kernel_size = [3, 3, 3] stride = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 0
    res = np.array([[[[[41.1337]]]],
        [[[[41.1783]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d4():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=1
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 1
    res = np.array([[[[[10.2772, 15.2571],
           [18.2521, 28.5663]],
          [[16.8695, 23.1947],
           [28.9226, 39.3930]]]],
        [[[[13.4966, 20.3464],
           [17.3102, 24.9590]],
          [[16.9992, 30.4095],
           [29.0104, 43.9873]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d5():
    """
    kernel_size = [3, 3, 3] stride = 2 padding=0 groups=3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 6
    kernel_size = [3, 3, 3]
    stride = 2
    padding = 0
    groups = 3
    res = np.array([[[[[12.5633]]],
         [[[12.5633]]],
         [[[16.0763]]],
         [[[16.0763]]],
         [[[12.4942]]],
         [[[12.4942]]]],
        [[[[14.2857]]],
         [[[14.2857]]],
         [[[15.0920]]],
         [[[15.0920]]],
         [[[11.8007]]],
         [[[11.8007]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d6():
    """
    kernel_size = [3, 3, 3] stride = 1 padding=1 w=0.7 b=-0.3
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    res = np.array([[[[[28.4936, 26.5148],
           [30.1473, 29.0135]],
          [[28.0812, 24.8464],
           [30.5865, 27.2751]]],
         [[[28.4936, 26.5148],
           [30.1473, 29.0135]],
          [[28.0812, 24.8464],
           [30.5865, 27.2751]]]],
        [[[[28.5248, 28.1234],
           [26.8961, 27.2252]],
          [[29.0086, 28.9991],
           [30.0757, 30.4911]]],
         [[[28.5248, 28.1234],
           [26.8961, 27.2252]],
          [[29.0086, 28.9991],
           [30.0757, 30.4911]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding,
             weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
             bias_attr=fluid.initializer.ConstantInitializer(value=-0.3))


def test_conv3d7():
    """
    kernel_size = [3, 3, 3] stride = 1 padding=1 w=0.7 b=-0.3 data_format="NDHWC"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4]).transpose(0, 2, 3, 4, 1)
    in_channels = 3
    out_channels = 2
    kernel_size = [3, 3, 3]
    stride = 1
    padding = 0
    data_format="NDHWC"
    res = np.array([[[[[28.4936, 26.5148],
           [30.1473, 29.0135]],
          [[28.0812, 24.8464],
           [30.5865, 27.2751]]],
         [[[28.4936, 26.5148],
           [30.1473, 29.0135]],
          [[28.0812, 24.8464],
           [30.5865, 27.2751]]]],
        [[[[28.5248, 28.1234],
           [26.8961, 27.2252]],
          [[29.0086, 28.9991],
           [30.0757, 30.4911]]],
         [[[28.5248, 28.1234],
           [26.8961, 27.2252]],
          [[29.0086, 28.9991],
           [30.0757, 30.4911]]]]]).transpose(0, 2, 3, 4, 1)
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, data_format=data_format,
             weight_attr=fluid.initializer.ConstantInitializer(value=0.7),
             bias_attr=fluid.initializer.ConstantInitializer(value=-0.3))


def test_conv3d8():
    """
    padding_mode = "reflect"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "reflect"
    res = np.array([[[[[31.212072, 36.005825, 32.827934, 37.62167],
          [38.86807, 44.092163, 42.229324, 47.45341]],
        [[35.650703, 38.057327, 34.95474, 37.361355],
          [41.854683, 44.12357, 39.39304, 41.661926]]]],
      [[[[45.362045, 42.906467, 44.43419, 41.978603],
          [36.93101, 38.189747, 37.721684, 38.98042]],
        [[40.621605, 42.058163, 46.969616, 48.406178],
          [42.471336, 43.393806, 43.987297, 44.90977]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d9():
    """
    padding_mode = "replicate"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "replicate"
    res = np.array([[[[[33.3041, 36.2519, 35.8157, 35.1824],
           [42.0793, 42.7396, 43.4696, 44.8771]],
          [[40.6510, 36.6411, 34.6294, 40.0582],
           [44.9131, 44.1236, 39.3930, 37.7209]]]],
        [[[[46.6887, 44.9393, 47.1106, 48.2136],
           [39.6648, 35.0459, 37.1553, 42.6259]],
          [[37.2624, 41.3674, 44.2589, 46.4076],
           [44.5598, 43.3938, 43.9873, 46.6693]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d10():
    """
    padding_mode = "circular"
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    padding_mode = "circular"
    res = np.array([[[[[39.0804, 37.8193, 37.7592, 40.6474],
           [41.5604, 43.3670, 40.9862, 43.4428]],
          [[40.7270, 40.2201, 38.0990, 42.4358],
           [40.1826, 44.1236, 39.3930, 42.4515]]]],
        [[[[44.2844, 39.8735, 44.3788, 44.9034],
           [45.4391, 39.5882, 41.8209, 44.7978]],
          [[44.2499, 42.8917, 46.3904, 46.5235],
           [45.1533, 43.3938, 43.9873, 46.0758]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d11():
    """
    padding_mode = "zeros" dilation = 2
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = 2
    padding_mode = "zeros"
    res = np.array([[[[[11.3474, 13.2012]]]],
        [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d12():
    """
    padding_mode = "zeros" dilation = [2, 2, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = [2, 2, 2]
    padding_mode = "zeros"
    res = np.array([[[[[11.3474, 13.2012]]]],
        [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d13():
    """
    padding_mode = "zeros" dilation = (2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = 1
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array([[[[[11.3474, 13.2012]]]],
        [[[[14.6407, 12.7313]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d14():
    """
    padding_mode = "zeros" dilation = (2, 2, 2)  padding = (1, 2, 2)
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = (1, 2, 2)
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array([[[[[14.4826, 8.5365, 14.4826, 8.5365],
           [14.4826, 8.5365, 14.4826, 8.5365]]]],
        [[[[14.1577, 11.3797, 14.1577, 11.3797],
           [14.1577, 11.3797, 14.1577, 11.3797]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))


def test_conv3d15():
    """
    padding_mode = "zeros" dilation = (2, 2)  padding = [1, 2]
    """
    np.random.seed(obj.seed)
    x = randtool("float", 0, 1, [2, 3, 4, 4, 4])
    in_channels = 3
    out_channels = 1
    kernel_size = [3, 3, 3]
    stride = [2, 2, 1]
    padding = [1, 2, 2]
    dilation = (2, 2, 2)
    padding_mode = "zeros"
    res = np.array([[[[[14.4826, 8.5365, 14.4826, 8.5365],
           [14.4826, 8.5365, 14.4826, 8.5365]]]],
        [[[[14.1577, 11.3797, 14.1577, 11.3797],
           [14.1577, 11.3797, 14.1577, 11.3797]]]]])
    obj.run(res=res, data=x, in_channels=in_channels, out_channels=out_channels,
             kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode,
             dilation=dilation,
             weight_attr=fluid.initializer.ConstantInitializer(value=1),
             bias_attr=fluid.initializer.ConstantInitializer(value=0))
