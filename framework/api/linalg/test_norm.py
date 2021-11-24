#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_norm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.gap = 1e-2
        self.delta = 1e-3
        # self.debug = True
        # self.static = False
        # self.dygraph = False
        # enable check grad
        # self.enable_backward = False


obj = TestNorm(paddle.linalg.norm)


def p_norm(x, axis, porder, keepdims=False):
    """
    compute norm
    """
    r = []
    if axis is None:
        x = x.flatten()
        if porder == np.inf:
            r = np.amax(np.abs(x))
        elif porder == -np.inf:
            r = np.amin(np.abs(x))
        else:
            r = np.linalg.norm(x, ord=porder)
    elif isinstance(axis, list or tuple) and len(axis) == 2:
        if porder == np.inf:
            axis = tuple(axis)
            r = np.amax(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == -np.inf:
            axis = tuple(axis)
            r = np.amin(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == 0:
            axis = tuple(axis)
            r = x.astype(bool)
            r = np.sum(r, axis)
        elif porder == 1:
            axis = tuple(axis)
            r = np.sum(np.abs(x), axis)
        else:
            axis = tuple(axis)
            xp = np.power(np.abs(x), porder)
            s = np.sum(xp, axis=axis, keepdims=keepdims)
            r = np.power(s, 1.0 / porder)
    else:
        if isinstance(axis, list):
            axis = tuple(axis)
        r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims).astype(x.dtype)

    return r


def frobenius_norm(x, axis=None, keepdims=False):
    """
    frobenius_norm
    """
    if isinstance(axis, list):
        axis = tuple(axis)
    if axis is None:
        axis = (-2, -1)
    r = np.linalg.norm(x, ord="fro", axis=axis, keepdims=keepdims).astype(x.dtype)
    return r


@pytest.mark.api_linalg_norm_vartype
def test_norm_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 4])
    res = [28.43878906]
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_norm_parameters
def test_norm():
    """
    default
    """
    np.random.seed(33)
    x = randtool("float", -10, 10, [3, 3, 3])
    res = [31.5736]
    obj.run(res=res, x=x, axis=None)


@pytest.mark.api_linalg_norm_parameters
def test_norm1():
    """
    ord = np.inf axis = [1, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = np.inf
    axis = [1, 2]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm2():
    """
    ord = np.inf axis = [1]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = np.inf
    axis = [1]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm3():
    """
    ord = np.inf axis = (1)
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = np.inf
    axis = 1
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm4():
    """
    ord = 0 axis = (1)
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 0
    axis = 1
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm5():
    """
    ord = -np.inf axis = (1)
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = -np.inf
    axis = 1
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm6():
    """
    ord = 1 axis = [1, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 1
    axis = [1, 2]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm7():
    """
    ord = 1 axis = [0, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 1
    axis = [0, 2]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm8():
    """
    ord = 2 axis = [0, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 2
    axis = [0, 2]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm9():
    """
    ord = 3 axis = [0, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 3
    axis = [0, 2]
    res = p_norm(x=x, axis=axis, porder=pord)
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm10():
    """
    ord = fro axis = None
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    pord = "fro"
    axis = None
    res = [16.8819]
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm11():
    """
    ord = fro axis = None BUG
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
    pord = "fro"
    axis = None
    res = [16.8819]
    obj.run(res=res, x=x, axis=axis, p=pord)


@pytest.mark.api_linalg_norm_parameters
def test_norm12():
    """
    ord = 3 axis = [0, 2]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    pord = 3
    axis = [0, 2]
    res = p_norm(x=x, axis=axis, porder=pord, keepdims=True)
    obj.run(res=res, x=x, axis=axis, p=pord, keepdim=True)
