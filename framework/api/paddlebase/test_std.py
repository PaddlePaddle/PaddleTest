#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test std
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestStd(APIBase):
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


obj = TestStd(paddle.std)


def ref_std(x, axis=None, unbiased=True, keepdim=False):
    """
    ref_std
    """
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.std(x, axis=axis, ddof=ddof, keepdims=keepdim)


@pytest.mark.api_base_std_vartype
def test_std_base():
    """
    base
    """
    x = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
    res = np.array([1.63299316])
    obj.base(res=res, x=x)


@pytest.mark.api_base_std_parameters
def test_std():
    """
    default
    """
    x = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
    res = [ref_std(x=x)]
    obj.run(res=res, x=x)


@pytest.mark.api_base_std_parameters
def test_std1():
    """
    default x rand
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = [ref_std(x=x)]
    obj.run(res=res, x=x)


@pytest.mark.api_base_std_parameters
def test_std2():
    """
    axis = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = 0
    res = ref_std(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_std_parameters
def test_std3():
    """
    axis = 0 unbiased=False
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = 0
    unbiased = False
    res = ref_std(x=x, axis=axis, unbiased=unbiased)
    obj.run(res=res, x=x, axis=axis, unbiased=unbiased)


@pytest.mark.api_base_std_parameters
def test_std4():
    """
    axis = [0, 1]
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = [0, 1]
    res = ref_std(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_std_parameters
def test_std5():
    """
    axis = (0, 1)
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = (0, 1)
    res = ref_std(x=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_std_parameters
def test_std6():
    """
    axis = (0, 1) keepdim=True
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    axis = (0, 1)
    keepdim = True
    res = ref_std(x=x, axis=axis, keepdim=keepdim)
    obj.run(res=res, x=x, axis=axis, keepdim=keepdim)
