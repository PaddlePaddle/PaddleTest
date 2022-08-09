#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test quantile
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare
import paddle
import pytest
import numpy as np


class TestQuantile(APIBase):
    """
    test quantile
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.static = False
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestQuantile(paddle.quantile)


@pytest.mark.api_base_quantile_vartype
def test_quantile_base():
    """
    base
    """
    q = 0.5
    axis = 0
    x = randtool("float", -5, 5, (3, 3, 3))
    exp = np.quantile(x, q=q, axis=axis)
    x_p = paddle.to_tensor(x)
    res = x_p.quantile(q=q, axis=axis)
    compare(res.numpy(), exp)


@pytest.mark.api_base_quantile_parameters
def test_quantile():
    """
    q = 0.75
    axis = 5
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    q = 0.75
    axis = 5
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    exp = np.quantile(x, q=q, axis=axis)
    x_p = paddle.to_tensor(x)
    res = x_p.quantile(q=q, axis=axis)
    compare(res.numpy(), exp)


@pytest.mark.api_base_quantile_parameters
def test_quantile1():
    """
    q = 0.75
    axis = 3
    keepdims = True
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    q = 0.75
    axis = 3
    keepdims = True
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    exp = np.quantile(x, q=q, axis=axis, keepdims=keepdims)
    x_p = paddle.to_tensor(x)
    res = x_p.quantile(q=q, axis=axis, keepdim=keepdims)
    compare(res.numpy(), exp)


@pytest.mark.api_base_quantile_parameters
def test_quantile2():
    """
    q = [0.25, 0.5, 0.75]
    axis = 3
    keepdims = False
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    q = [0.25, 0.5, 0.75]
    axis = 3
    keepdims = False
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    exp = np.quantile(x, q=q, axis=axis, keepdims=keepdims)
    x_p = paddle.to_tensor(x)
    res = x_p.quantile(q=q, axis=axis, keepdim=keepdims)
    compare(res.numpy(), exp)


@pytest.mark.api_base_quantile_parameters
def test_quantile3():
    """
    q = (0.11, 0.5, 0.73, 0.9)
    axis = 4
    keepdims = False
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    q = (0.11, 0.5, 0.73, 0.9)
    axis = 4
    keepdims = False
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    exp = np.quantile(x, q=q, axis=axis, keepdims=keepdims)
    x_p = paddle.to_tensor(x)
    res = x_p.quantile(q=q, axis=axis, keepdim=keepdims)
    compare(res.numpy(), exp)


@pytest.mark.api_base_quantile_parameters
def test_quantile4():
    """
    q = 0.5
    axis = None
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    paddle.disable_static()
    q = 0.5
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    x_p = paddle.to_tensor(x)
    exp = np.quantile(x, q=q)
    res = x_p.quantile(q=q)
    compare(res.numpy(), exp)
