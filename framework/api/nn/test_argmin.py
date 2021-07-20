#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test argmin
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestArgmin(APIBase):
    """
    test argmin
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8]
        # self.debug = True
        # enable check grad
        self.no_grad_var = {"x", "axis"}
        self.enable_backward = False


obj = TestArgmin(paddle.argmin)


@pytest.mark.api_base_argmin_vartype
def test_argmin_base():
    """
    base,axis = 0,dtype = np.int32,keepdim = True
    """
    x = randtool('float', -10, 10, [3, 3])
    axis = 0
    dtype = np.int32
    keepdim = False
    res = np.argmin(a=x, axis=axis)
    obj.base(res=res, x=x, axis=axis, dtype=dtype, keepdim=keepdim)


# test axis
@pytest.mark.api_base_argmin_parameters
def test_argmin():
    """
    default
    """
    x = randtool('float', -10, 10, [3, 3, 3])
    res = np.argmin(a=x)
    obj.run(res=[res], x=x)


@pytest.mark.api_base_argmin_parameters
def test_argmin1():
    """
    axis >0,dtype = np.int64
    """
    x = randtool('float', -10, 10, [3, 3, 5])
    axis = 2
    dtype = np.int64
    res = np.argmin(a=x, axis=axis)
    obj.run(res=res, x=x, axis=axis, dtype=dtype)


@pytest.mark.api_base_argmin_parameters
def test_argmin2():
    """
    axis <0,dtype = 'int64'
    """
    x = randtool('float', -10, 10, [3, 3, 2, 1])
    axis = -1
    dtype = 'int64'
    res = np.argmin(a=x, axis=axis)
    obj.run(res=res, x=x, axis=axis, dtype=dtype)


@pytest.mark.api_base_argmin_parameters
def test_argmin3():
    """
    dtype = 'int32'
    """
    x = randtool('int', -10, 10, [3, 3])
    dtype = 'int32'
    res = np.argmin(a=x)
    obj.run(res=[res], x=x, dtype=dtype)


@pytest.mark.api_base_argmin_parameters
def test_argmin4():
    """
    axis = -R
    """
    x = randtool('int', -100, 10, [3, 3])
    axis = -2
    res = np.argmin(a=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_argmin_parameters
def test_argmin5():
    """
    0<axis<R,keepdim = True
    """
    x = np.array([0, 1, 2])
    axis = -1
    keepdim = True
    res = np.argmin(a=x, axis=axis)
    obj.run(res=[res], x=x, axis=axis, keepdim=keepdim)


@pytest.mark.api_base_argmin_parameters
def test_argmin6():
    """
    keepdim = False
    """
    x = randtool('float', -1, 1, [3, 3, 4])
    axis = 1
    keepdim = False
    res = np.argmin(a=x, axis=axis)
    obj.run(res=res, x=x, axis=axis, keepdim=keepdim)


@pytest.mark.api_base_argmin_parameters
def test_argmin7():
    """
    keepdim = None
    """
    x = np.array([[-1], [2], [3]])
    keepdim = None
    res = np.argmin(a=x)
    obj.run(res=[res], x=x, keepdim=keepdim)


@pytest.mark.api_base_argmin_parameters
def test_argmin8():
    """
    x_shape=[3,2]
    """
    x = np.array([[7, 10, 9], [-100, 2, -10]])
    axis = -1
    res = np.argmin(a=x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


# exception case
@pytest.mark.api_base_argmin_exception
def test_argmin9():
    """
    axis = R,c++error
    """
    x = randtool('float', -1, 1, [3, 3])
    axis = 2
    obj.exception(mode='c', etype='InvalidArgumentError', x=x, axis=axis)


@pytest.mark.api_base_argmin_exception
def test_argmin10():
    """
    axis = float
    """
    x = np.array([[-1], [2], [3]])
    axis = float(1.0)
    obj.exception(mode='python', etype=TypeError, x=x, axis=axis)


@pytest.mark.api_base_argmin_exception
def test_argmin11():
    """
    dtype = float32.c++error
    """
    x = randtool('float', -10, 10, [3, 3])
    dtype = np.float32
    obj.exception(mode='c', etype='InvalidArgumentError', x=x, dtype=dtype)


@pytest.mark.api_base_argmin_exception
def test_argmin12():
    """
    dtype = None
    """
    x = randtool('int', -100, 10, [3, 3])
    dtype = None
    obj.exception(mode='python', etype=ValueError, x=x, dtype=dtype)
