#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_maxout
"""
import math
from apibase import APIBase
from apibase import randtool
from apibase import tanh
import paddle
import pytest
import numpy as np


class TestFunctionalMaxout(APIBase):
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
        # self.delta = 1e-1


obj = TestFunctionalMaxout(paddle.nn.functional.maxout)


@pytest.mark.api_nn_Maxout_vartype
def test_maxout_base():
    """
    base
    """
    x = randtool("float", -10, 10, [100, 4, 3, 3])
    # x = np.random.rand(1, 2, 2, 2)
    # print(x)
    groups = 2
    res = []
    for i in range(x.shape[1] // groups):
        r = np.max(x[:, i * groups : (i + 1) * groups], axis=1)
        res.append(r)
    # print(res)
    res = np.transpose(np.array(res), (1, 0, 2, 3))
    obj.base(res=res, x=x, groups=2)


@pytest.mark.api_nn_Maxout_parameters
def test_maxout():
    """
     default axis = 1
    """
    x = randtool("float", -10, 10, [10, 9, 3, 3])
    # print(x)
    groups = 3
    res = []
    for i in range(x.shape[1] // groups):
        r = np.max(x[:, i * groups : (i + 1) * groups], axis=1)
        res.append(r)
    # print(res)
    res = np.transpose(np.array(res), (1, 0, 2, 3))
    obj.run(res=res, x=x, groups=groups)


@pytest.mark.api_nn_Maxout_parameters
def test_maxout1():
    """
    axis = -1 or 3
    """
    x = randtool("float", 0, 255, [9, 2, 2, 6])
    # print(x)
    groups = 2
    res = []
    y = np.transpose(x, (0, 3, 1, 2))
    for i in range(y.shape[1] // groups):
        r = np.max(y[:, i * groups : (i + 1) * groups], axis=1)
        res.append(r)
    # print(res)
    res = np.transpose(np.array(res), (1, 0, 2, 3))
    res = np.transpose(res, (0, 2, 3, 1))
    # obj.run(res=res, data=x, groups=groups, axis=-1)
    obj.run(res=res, x=x, groups=groups, axis=3)


@pytest.mark.api_nn_Maxout_exception
def test_maxout2():
    """
    groups = 1
    """
    x = randtool("float", -10, 10, [10, 9, 3, 3])
    # print(x)
    groups = 1
    res = []
    for i in range(x.shape[1] // groups):
        r = np.max(x[:, i * groups : (i + 1) * groups], axis=1)
        res.append(r)
    # print(res)
    # res = np.transpose(np.array(res), (1, 0, 2, 3))
    obj.exception(etype=ValueError, mode="python", x=x, groups=groups)


@pytest.mark.api_nn_Maxout_exception
def test_maxout3():
    """
    connot divisible:
    channel = 9
    groups = 2
    """
    x = randtool("float", -10, 10, [1, 9, 3, 3])
    # print(x)
    groups = 2
    res = []
    for i in range(x.shape[1] // groups):
        r = np.max(x[:, i * groups : (i + 1) * groups], axis=1)
        res.append(r)
    # print(res)
    # res = np.transpose(np.array(res), (1, 0, 2, 3))
    obj.exception(etype=ValueError, mode="python", x=x, groups=groups)
