#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test sqrt
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSqrt(APIBase):
    """
    test sqrt
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.float16]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestSqrt(paddle.sqrt)
obj2 = TestSqrt(paddle.sqrt)
obj2.enable_backward = False


@pytest.mark.api_base_sqrt_vartype
def test_sqrt_base():
    """
    default
    """
    x = np.array([1, 2, 3])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt():
    """
    x = np.array([0.9, 0.8, 0.7, 0.6])
    """
    x = np.array([0.9, 0.8, 0.7, 0.6])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt1():
    """
    x_data_type=3-D tensor
    """
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt2():
    """
    x = np.array([0.9])
    """
    x = np.array([0.9])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt3():
    """
    x = np.array([-2, -3])
    """
    x = np.array([-2, -3])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


# def test_sqrt4():
#     """
#     x = np.array([])=>[]?none,AssertionError
#     """
#     x = np.array([])
#     res = np.sqrt(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt5():
    """
    x = np.array([1, 1e+1])
    """
    x = np.array([1, 1e1])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt6():
    """
    name
    """
    x = np.array([3, 100, 100])
    name = None
    res = np.sqrt(x)
    obj.run(res=res, x=x, name=name)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt7():
    """
    x = np.array([-1])=>[nan]
    """
    x = np.array([-1])
    res = np.sqrt(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sqrt_parameters
def test_sqrt8():
    """
    x = np.array([0])=>[inf]
    """
    x = np.array([0])
    res = np.sqrt(x)
    obj2.run(res=res, x=x)


@pytest.mark.api_base_sqrt_exception
def test_sqrt9():
    """
    x = [-3, 2],c++ error
    """
    x = [-3, 2]
    obj.exception(mode="c", etype="InvalidArgumentError", x=x)
