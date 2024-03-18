#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_leakyrelu
"""
import math
from apibase import APIBase
from apibase import randtool
from apibase import tanh
import paddle
import pytest
import numpy as np


class TestLeakyReLU(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestLeakyReLU(paddle.nn.LeakyReLU)


@pytest.mark.api_nn_LeakyReLU_vartype
def test_leakyrelu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] >= 0:
            res.append(x.flatten()[i])
        else:
            res.append(0.01 * x.flatten()[i])
    res = np.array(res).reshape(x.shape)
    obj.base(res=res, data=x)


@pytest.mark.api_nn_leakyrelu_parameters
def test_leakyrelu():
    """
    default negative_slope = 0.01
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] >= 0:
            res.append(x.flatten()[i])
        else:
            res.append(0.01 * x.flatten()[i])
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, data=x)


@pytest.mark.api_nn_leakyrelu_parameters
def test_leakyrelu1():
    """
    default negative_slope = 0 --->>> ReLU
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # 算法
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] >= 0:
            res.append(x.flatten()[i])
        else:
            res.append(0)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, data=x, negative_slope=0)


@pytest.mark.api_nn_leakyrelu_parameters
def test_leakyrelu2():
    """
    default negative_slope < 0
    negative_slope = -1 --->>> y=|x|
    """
    x = randtool("float", -10, 10, [10, 3, 3])
    # 算法
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] >= 0:
            res.append(x.flatten()[i])
        else:
            res.append(-x.flatten()[i])
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, data=x, negative_slope=-1)
