#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test divide
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare

import paddle
import pytest
import numpy as np


class TestDivide(APIBase):
    """
    test normal
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestDivide(paddle.divide)


@pytest.mark.api_base_divide_vartype
def test_divide_base():
    """
    base,x_shape=y_shape,np.int32, np.int64 error
    """
    x = np.array([2, 3, 4])
    y = np.array([1, 5, 2])
    res = np.true_divide(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide():
    """
    x_shape>y_shape
    """
    x = randtool("float", 1, 10, [3, 3, 3])
    y = randtool("float", 1, 10, [3])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide1():
    """
    x_shape<y_shape
    """
    x = randtool("float", 1, 10, [3])
    y = randtool("float", -1, 10, [3, 1])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide2():
    """
    x=y!=0
    """
    x = np.array([-1e-1, 2e1])
    y = np.array([-1e-1, 2e1])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide3():
    """
    x=0,y!=0
    """
    x = np.zeros([3])
    y = np.array([1, 2, 3])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide4():
    """
    x<0,name = None
    """
    x = np.array([[-3], [-11], [-2]])
    y = np.array([[-1], [2], [1]])
    name = None
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y, name=name)


@pytest.mark.api_base_divide_parameters
def test_divide5():
    """
    x_shape<y_shape
    """
    x = randtool("float", 1, 10, [3, 1])
    y = randtool("float", -1, 10, [3, 2])
    name = ""
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y, name=name)


obj.enable_backward = False


@pytest.mark.api_base_divide_parameters
def test_divide7():
    """
    x=y=0
    """
    x = np.zeros([3])
    y = np.zeros([3])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_divide_parameters
def test_divide8():
    """
    x!=0,y=0
    """
    x = np.array([1, 2, 3])
    y = np.zeros([3])
    res = np.true_divide(x, y)
    obj.run(res=res, x=x, y=y)


# @pytest.mark.api_base_divide_parameters
# def test_divide9():
#     """
#     res=paddle.to_tensor(99.0) / 100
#     exp=paddle.divide(paddle.to_tensor(99.0), paddle.to_tensor(100.0))
#     """
#     paddle.disable_static()
#     x = np.ones((2, 3, 3, 1, 2, 5, 2)).astype(np.float32) * 99
#     y = np.ones((2, 3, 3, 1, 2, 5, 2)).astype(np.float32) * 100
#     res = paddle.to_tensor(x) / 100
#     exp = paddle.divide(paddle.to_tensor(x), paddle.to_tensor(y))
#     compare(exp.numpy(), res.numpy(), delta=1e-10, rtol=1e-10)
