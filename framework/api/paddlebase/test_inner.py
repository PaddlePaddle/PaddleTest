#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_inner
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestInner(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-2


obj = TestInner(paddle.inner)


def cal_inner(x, y):
    """
    calculate inner
    """
    shape0, shape1 = x.shape, y.shape
    batchsize0 = np.product(shape0) // (shape0[-1])
    batchsize1 = np.product(shape1) // (shape1[-1])
    x_t = x.reshape(-1, shape0[-1])
    y_t = y.reshape(-1, shape1[-1])
    tmp = []
    for i in range(batchsize0):
        for j in range(batchsize1):
            tmp.append(np.dot(x_t[i], y_t[j]))
    return np.array(tmp).reshape(shape0[:-1] + shape1[:-1])


@pytest.mark.api_base_inner_vartype
def test_inner_base():
    """
    base
    """
    x = randtool("float", -4, 4, (4, 4))
    y = randtool("float", -2, 2, (4, 4))
    res = cal_inner(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner0():
    """
    default
    """
    x = randtool("float", -4, 4, (4,))
    y = randtool("float", -2, 2, (4,))
    res = np.array([cal_inner(x, y)])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner1():
    """
    x, y: 2d-tensor
    """
    x = randtool("float", -4, 4, (3, 4))
    y = randtool("float", -2, 2, (5, 4))
    res = cal_inner(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner2():
    """
    x, y: 3d-tensor
    """
    x = randtool("float", -4, 4, (5, 3, 4))
    y = randtool("float", -2, 2, (2, 5, 4))
    res = cal_inner(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner3():
    """
    x, y: 4d-tensor
    """
    x = randtool("float", -4, 4, (2, 5, 3, 4))
    y = randtool("float", -2, 2, (3, 2, 5, 4))
    res = cal_inner(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner4():
    """
    x: 2d-tensor, y: 3d-tensor
    """
    x = randtool("float", -4, 4, (3, 4))
    y = randtool("float", -2, 2, (3, 2, 4))
    res = cal_inner(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_inner_parameters
def test_inner5():
    """
    x: 2d-tensor, y: 4d-tensor
    """
    x = randtool("float", -4, 4, (3, 4))
    y = randtool("float", -2, 2, (3, 2, 5, 4))
    res = cal_inner(x, y)
    obj.run(res=res, x=x, y=y)
