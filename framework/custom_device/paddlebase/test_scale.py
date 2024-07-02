#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test scale
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestScale(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestScale(paddle.scale)


@pytest.mark.api_base_scale_vartype
def test_scale_base():
    """
    base
    """
    # x = -1 + 2 * np.random.random(size=[3, 3, 3])
    x = randtool("int", -10, 10, [3, 3, 3])
    scale = 1.0
    bias = 0.0
    bias_after_scale = True
    act = None
    res = x * scale + bias
    obj.base(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)


@pytest.mark.api_base_scale_parameters
def test_scale1():
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    scale = 2.0
    bias = 4.0
    bias_after_scale = True
    act = None
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    scale = 2.0
    bias = 4.0
    bias_after_scale = True
    act = None
    res = x * scale + bias
    obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)


@pytest.mark.api_base_scale_parameters
def test_scale2():
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    scale = 2.0
    bias = 4.0
    bias_after_scale = False
    act = None
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    scale = 2.0
    bias = 4.0
    bias_after_scale = False
    act = None
    res = (x + bias) * scale
    obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)


# @pytest.mark.api_base_scale_parameters
# def test_scale3():
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = 'softmax'
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = "softmax"
#     res = np.array(
#         [
#             [
#                 [14.27946932, -0.1367496, 15.71402881],
#                 [17.02113557, 15.53676553, 23.37986002],
#                 [-0.22910936, 0.67412584, 6.58322372],
#             ],
#             [
#                 [-8.6544309, 4.74657431, 0.98828855],
#                 [-8.78491276, -0.6906711, -0.76046861],
#                 [22.37076412, -7.10690386, -12.1887529],
#             ],
#             [
#                 [8.59181793, -6.90934915, 10.35677566],
#                 [-9.17215801, -9.94048039, 12.7021191],
#                 [5.20851438, 1.33105715, 18.83352911],
#             ],
#         ]
#     )
#     obj.static = False
#     obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)
#     obj.static = True
#
#
# @pytest.mark.api_base_scale_parameters
# def test_scale4():
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = 'tanh'
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = "tanh"
#     res = np.array(
#         [[[13.13519313 , 16.42913885 , 17.08653224],
#           [-3.20290876 , 18.90116387 , - 9.6769234],
#           [1.69099792  ,23.3178677   , - 14.77466926]],
#          [[-7.45071035 , 23.15734955 , - 12.72081163],
#           [-12.24417866,   4.70098812, - 7.22207556],
#           [-10.58582287,  18.71731506,  12.3668466]],
#          [[10.70028107 , 17.17056613 ,  6.70880192],
#           [-5.43916291 ,- 5.53052148 , - 8.34774478],
#           [8.74493502  ,23.74025896  , - 6.02444433]]]
#     )
#     obj.static = False
#     obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)
#     obj.static = True
#
#
# @pytest.mark.api_base_scale_parameters
# def test_scale5():
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = 'sigmoid'
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = "sigmoid"
#     res = np.array(
#         [[[-2.4812946  , 9.92731226   ,3.77018558],
#           [-12.44055776,- 7.27950665  , 0.64194158],
#           [8.52167936  ,- 2.92359916  ,20.2925092]],
#          [[20.69192489 ,11.08189683   ,6.90653356],
#           [-13.34207556,- 2.74248474  , 3.1176005],
#           [12.99787298 , - 13.23774121,   14.81803512]],
#          [[0.87175305  ,6.29756237    , 18.04124564],
#           [8.57679749  ,- 10.19730374 , 18.65538271],
#           [10.78978112 , 11.57030431  , 14.2974599]]]
#     )
#     obj.static = False
#     obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)
#     obj.static = True
#
#
# @pytest.mark.api_base_scale_parameters
# def test_scale6():
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = 'relu'
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     scale = 2.0
#     bias = 4.0
#     bias_after_scale = True
#     act = "relu"
#     res = np.array(
#         [[[22.82150458 ,- 1.44266708  ,10.94878543],
#           [11.24573921 ,- 8.11563283  ,17.63086559],
#           [5.59851295  ,10.68863262   ,11.88557438]],
#          [[-3.30251057 ,  3.79809964  , 23.14695339],
#           [10.04161452 ,- 13.64713773 , - 15.59077411],
#           [8.73318017  ,- 7.30874951  , 20.7254908]],
#          [[9.16791892  , 4.41790108   , 14.49498511],
#           [-10.85745166, - 11.04512934,  17.652719],
#           [-11.51011394, - 3.46613062 , 18.26800255]]]
#     )
#     obj.static = False
#     obj.run(res=res, x=x, scale=scale, bias=bias, bias_after_scale=bias_after_scale, act=act)
#     obj.static = True
