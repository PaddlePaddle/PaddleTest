#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_reshape_
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestReshape(APIBase):
    """
    test reshape_
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        self.enable_backward = False
        self.no_grad_var = ["shape"]


obj = TestReshape(paddle.reshape_)


@pytest.mark.api_base_reshape__vartype
def test_reshape__base():
    """
    base
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [1, 4]
    res = np.reshape(x, shape)
    obj.base(res=res, x=x, shape=shape)


@pytest.mark.api_base_gather_parameters
def test_reshape_1():
    """
    shape has -1
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [1, -1]
    res = np.reshape(x, shape)
    obj.base(res=res, x=x, shape=shape)


@pytest.mark.api_base_gather_exception
def test_reshape_2():
    """
    shape has more than one -1
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [-1, -1]
    obj.exception(mode="c", etype="InvalidArgument", x=x, shape=shape)


@pytest.mark.api_base_gather_exception
def test_reshape_3():
    """
    shape has negtive_num
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [-2, 2]
    obj.exception(mode="c", etype="InvalidArgument", x=x, shape=shape)


@pytest.mark.api_base_gather_parameters
def test_reshape_4():
    """
    shape has 0
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [2, 0]
    res = np.reshape(x, [2, 2])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_gather_exception
def test_reshape_5():
    """
    num of 0 is more than dim
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [0, 0, 0]
    obj.exception(mode="c", etype="InvalidArgument", x=x, shape=shape)


@pytest.mark.api_base_gather_exception
def test_reshape_6():
    """
    shape larger than input
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [3, 4]
    obj.exception(mode="c", etype="InvalidArgument", x=x, shape=shape)


@pytest.mark.api_base_gather_exception
def test_reshape_7():
    """
    shape smaller than input
    """
    x = np.array([[8, 4], [7, 9]])
    shape = [1, 1]
    obj.exception(mode="c", etype="InvalidArgument", x=x, shape=shape)


@pytest.mark.api_base_gather_parameters
def test_reshape_8():
    """
    shape is tuple
    """
    x = np.array([[8, 4], [7, 9]])
    shape = (1, 4)
    res = np.reshape(x, shape)
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_gather_parameters
def test_reshape_9():
    """
    shape is tensor
    """
    x = np.array([[8, 4], [7, 9]])
    shape = np.array([1, 4]).astype("int32")
    res = np.reshape(x, (1, 4))
    obj.run(res=res, x=x, shape=shape)
