#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_expand_as.py
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestExpandAs(APIBase):
    """
    test expand_as
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.no_grad_var = {"x", "y"}
        # self.enable_backward = False


obj = TestExpandAs(paddle.expand_as)


@pytest.mark.api_base_expand_as_vartype
def test_expand_as_base():
    """
    base,y_shape=[3,3]
    """
    x = np.array([1.1, 2, 3])
    y = np.ones([3, 3])
    res = np.array([[1.1, 2, 3], [1.1, 2, 3], [1.1, 2, 3]])
    obj.base(res=res, x=x, y=y)


obj.enable_backward = False


@pytest.mark.api_base_expand_as_parameters
def test_expand_as1():
    """
    y_shape=[3,3],x_type=np.int32
    """
    x = np.array([1, 2, 3]).astype(np.int32)
    y = np.ones([3, 3]).astype(np.int32)
    res = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as2():
    """
    y_shape=[3,3],x_type=np.int64
    """
    x = np.array([[1], [2], [3]]).astype(np.int64)
    y = np.ones([3, 2]).astype(np.int64)
    res = np.array([[1, 1], [2, 2], [3, 3]])
    obj.run(res=res, x=x, y=y)


# def test_expand_as3():
#     """
#     x_type=np.bool,todo:fix RC
#     """
#     x = np.array([False, True]).astype(np.bool)
#     y = np.ones([1, 2]).astype(np.bool)
#     res = np.array([[False, True]])
#     obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as4():
    """
    y_shape=x_shape
    """
    x = np.array([1]).astype("int32")
    y = np.ones([1]).astype(np.int32)
    res = x
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_vartype
def test_expand_as5():
    """
    x_type='int64'
    """
    x = np.array([1, 2, 3]).astype("int64")
    y = np.ones([2, 3]).astype(np.int64)
    res = np.array([[1, 2, 3], [1, 2, 3]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as7():
    """
    y_shape = 2-D
    """
    x = np.array([1, 2]).astype(np.int32)
    y = np.ones([2, 1, 2]).astype(np.int32)
    res = np.array([[[1, 2]], [[1, 2]]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as8():
    """
    y_shape = 6
    """
    x = np.array([1]).astype(np.int64)
    y = np.ones([1, 1, 1, 1, 1, 1]).astype(np.int64)
    res = np.array([[[[[[1]]]]]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as9():
    """
    Y_shape = [1],x_shape=shape_size
    """
    x = np.array([1]).astype(np.int32)
    y = np.ones([1]).astype(np.int32)
    res = np.array([1])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_expand_as_parameters
def test_expand_as12():
    """
    y_shape = 2-D
    """
    x = np.array([[1, 2], [2, 3]]).astype(np.int32)
    y = np.ones([1, 2, 2]).astype(np.int32)
    res = np.array([[[1, 2], [2, 3]]])
    obj.run(res=res, x=x, y=y)
