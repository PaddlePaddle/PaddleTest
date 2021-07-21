#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test var
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestVar(APIBase):
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
        # self.dygraph = False
        # enable check grad
        # self.enable_backward = False


obj = TestVar(paddle.var)


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    """
    ref_var
    """
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


@pytest.mark.api_base_var_vartype
def test_var_base():
    """
    base
    """
    x = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
    res = np.array([2.66666667])
    obj.base(res=res, x=x)


@pytest.mark.api_base_var_parameters
def test_var():
    """
    default
    """
    x = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
    res = np.array([2.66666667])
    obj.run(res=res, x=x)


@pytest.mark.api_base_var_parameters
def test_var1():
    """
    default x rand
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = [ref_var(x=x)]
    obj.run(res=res, x=x)


@pytest.mark.api_base_var_parameters
def test_var2():
    """
    axis = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = 0
    res = ref_var(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_var_parameters
def test_var3():
    """
    axis = 0 unbiased=False
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = 0
    unbiased = False
    res = ref_var(x=x, axis=axis, unbiased=unbiased)
    obj.run(res=res, x=x, axis=axis, unbiased=unbiased)


@pytest.mark.api_base_var_parameters
def test_var4():
    """
    axis = [0, 1]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [0, 1]
    res = ref_var(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_var_parameters
def test_var5():
    """
    axis = (0, 1)
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = (0, 1)
    res = ref_var(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_var_parameters
def test_var6():
    """
    axis = (0, 1) keepdim=True
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = (0, 1)
    keepdim = True
    res = ref_var(x=x, axis=axis, keepdim=keepdim)
    obj.run(res=res, x=x, axis=axis, keepdim=keepdim)
