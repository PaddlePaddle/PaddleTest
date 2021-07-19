#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test abs
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAbs(APIBase):
    """
    test abs
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestAbs(paddle.abs)
obj1 = TestAbs(paddle.abs)
obj1.types = [np.float16, np.int32]


@pytest.mark.api_base_abs_vartype
def test_abs_base():
    """
    base
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    res = np.abs(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs():
    """
    default
    """
    x = np.array([-1, -2, -3, 4])
    res = np.abs(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs1():
    """
    x>0
    """
    x = randtool("float", 1, 10, (3, 3, 3))
    res = np.abs(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs2():
    """
    x<0
    """
    x = randtool("float", -100, 0, (3, 3, 3))
    res = np.abs(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs3():
    """
    x=0
    """
    x = np.ones([5, 3])
    res = np.abs(x)
    obj.run(res=res, x=x)


# def test_abs4():
#     """
#     x = np.array([])=>[]?none,AssertionError
#     """
#     x = np.array([])
#     res = np.abs(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs5():
    """
    x = np.array([1e+10])
    """
    x = np.array([1e10])
    res = np.abs(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_abs_parameters
def test_abs6():
    """
    x = np.array([1e-10])
    """
    x = np.array([1e-10])
    res = np.abs(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_abs_exception
def test_abs7():
    """
    x = [-3, 2],c++ error
    """
    x = [-3, 2]
    obj.exception(mode="c", etype="InvalidArgumentError", x=x)
