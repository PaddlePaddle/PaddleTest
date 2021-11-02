#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_normalize
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFuntionalNormalize(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        # should support fp64, fp64 waitting for fix.
        self.types = [np.float32, np.float64]
        # self.rtol = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False


obj = TestFuntionalNormalize(paddle.nn.functional.normalize)


def cal_normalize(x, axis=1, p=2):
    """
    calculate normalize api
    """
    shape = x.shape
    norm_reshape = list(shape)
    norm_reshape[axis] = 1
    norm = np.linalg.norm(x, ord=p, axis=axis)
    divisor = []
    for item in norm.flatten():
        if item > 1e-12:
            divisor.append(item)
        else:
            divisor.append(1e-12)
    divisor = np.array(divisor).reshape(norm_reshape)
    return x / divisor


@pytest.mark.api_nn_normalize_vartype
def test_normalize_base():
    """
    base
    x: 2-D tensor
    """
    x = randtool("float", -0, 1, (2, 4))
    res = cal_normalize(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_normalize_parameters
def test_normalize0():
    """
    default
    x: 2-D tensor
    """
    x = randtool("float", -0, 5, (3, 4))
    res = cal_normalize(x)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_normalize_parameters
def test_normalize1():
    """
    default
    x: 1-D tensor
    axis should be set 0
    """
    x = randtool("float", -0, 5, (4,))
    res = cal_normalize(x, axis=0)
    obj.run(res=res, x=x, axis=0)


@pytest.mark.api_nn_normalize_parameters
def test_normalize2():
    """
    default
    x: 3-D tensor
    """
    x = randtool("float", -0, 5, (4, 5, 6))
    res = cal_normalize(x)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_normalize_parameters
def test_normalize3():
    """
    default
    x: 4-D tensor
    """
    x = randtool("float", -0, 5, (4, 5, 6, 7))
    res = cal_normalize(x)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_normalize_parameters
def test_normalize4():
    """
    x: 4-D tensor
    p = 1
    """
    x = randtool("float", -0, 5, (4, 5, 6, 7))
    res = cal_normalize(x, p=1)
    obj.run(res=res, x=x, p=1)


@pytest.mark.api_nn_normalize_parameters
def test_normalize5():
    """
    x: 4-D tensor
    p = 4
    """
    x = randtool("float", -0, 5, (4, 5, 6, 7))
    res = cal_normalize(x, p=4)
    obj.run(res=res, x=x, p=4)


@pytest.mark.api_nn_normalize_parameters
def test_normalize6():
    """
    x: 4-D tensor
    p = 4
    axis = 3
    """
    x = randtool("float", -0, 5, (4, 5, 6, 7))
    res = cal_normalize(x, p=4, axis=3)
    obj.run(res=res, x=x, p=4, axis=3)


@pytest.mark.api_nn_normalize_parameters
def test_normalize7():
    """
    x: 4-D tensor
    p = 4
    axis = -2
    """
    x = randtool("float", -0, 5, (4, 5, 6, 7))
    res = cal_normalize(x, p=4, axis=-2)
    obj.run(res=res, x=x, p=4, axis=-2)


@pytest.mark.api_nn_normalize_parameters
def test_normalize7():
    """
    x: 2-D tensor
    p = 1.2 (float)
    axis = -2
    """
    np.random.seed(22)
    x = np.random.rand(4, 2)
    res = np.array(
        [[0.33371879, 0.77111008], [0.36451978, 0.74473367], [0.37261313, 0.73769581], [0.30973973, 0.79119033]]
    )
    obj.run(res=res, x=x, p=1.2)
