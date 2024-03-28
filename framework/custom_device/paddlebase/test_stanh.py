#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test stanh
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import pytest


class TestStanh(APIBase):
    """
    test stanh
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestStanh(paddle.stanh)


def naive_stanh(x, scale_a=0.67, scale_b=1.72):
    """
    naive_stanh for test
    """
    out = scale_b * np.tanh(scale_a * x)
    return out


@pytest.mark.api_base_stanh_vartype
def test_stanh_base():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float64)
    scale_a = 0.67
    scale_b = 1.72
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.base(res=res, x=x, scale_a=scale_a, scale_b=scale_b)


@pytest.mark.api_base_stanh_parameters
def test_stanh_1():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float64)
    scale_a = 0.67
    scale_b = 1.72
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.base(res=res, x=x, scale_a=scale_a, scale_b=scale_b)


@pytest.mark.api_base_stanh_parameters
def test_stanh_2():
    """
    x.shape=(1, 2)
    float32
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float32)
    scale_a = 1.43
    scale_b = 4.56
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.base(res=res, x=x, scale_a=scale_a, scale_b=scale_b)


@pytest.mark.api_base_stanh_parameters
def test_stanh_3():
    """
    x.shape=(2, 2)
    float64
    """
    x = randtool("float", -10, 10, [2, 2]).astype(np.float64)
    scale_a = 6.42
    scale_b = 3.58
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.base(res=res, x=x, scale_a=scale_a, scale_b=scale_b)


@pytest.mark.api_base_stanh_parameters
def test_stanh_4():
    """
    x.shape=(1, )
    """
    x = np.array([8.931])
    scale_a = 0.67
    scale_b = 1.72
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.base(res=res, x=x, scale_a=scale_a, scale_b=scale_b)


@pytest.mark.api_base_stanh_parameters
def test_stanh_5():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", -10, 10, [2, 3, 2, 2]).astype(np.float32)
    scale_a = 0.67
    scale_b = 1.72
    res = naive_stanh(x, scale_a=scale_a, scale_b=scale_b)
    obj.run(res=res, x=x, scale_a=scale_a, scale_b=scale_b)
