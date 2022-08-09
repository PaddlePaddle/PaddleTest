#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sum
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSum(APIBase):
    """
    test sum
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = True


obj = TestSum(paddle.sum)


@pytest.mark.api_base_sum_vartype
def test_sum_base():
    """
    base
    axis=None
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = [np.sum(x)]
    print("res:{}".format(res))
    obj.base(res=res, x=x)


@pytest.mark.api_base_sum_parameters
def test_sum1():
    """
    axis<rank(x)
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = 1
    res = np.sum(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_sum_parameters
def test_sum2():
    """
    axis>rank(x)
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = 3
    obj.exception(mode="c", etype="InvalidArgumentError", x=x, axis=axis)


@pytest.mark.api_base_sum_parameters
def test_sum3():
    """
    axis=negtive num
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = -1
    res = np.sum(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_sum_parameters
def test_sum4():
    """
    axis=list
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = [0, 1]
    res = [2.8]
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_sum_parameters
def test_sum5():
    """
    axis=tuple
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = (0, 1)
    res = [2.8]
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_sum_parameters
def test_sum6():
    """
    dtype=float64
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = [np.sum(x)]
    obj.base(res=res, x=x, dtype="float64")


@pytest.mark.api_base_sum_parameters
def test_sum7():
    """
    keepdim=True
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = np.array([[1.5, 1.3]])
    obj.base(res=res, x=x, axis=0, keepdim=True)


@pytest.mark.api_base_sum_parameters
def test_sum8():
    """
    keepdim=True
    """
    x = np.random.uniform(-1, 1, [2, 3, 4, 5])
    res = np.sum(x).reshape((1, 1, 1, 1))
    obj.run(res=res, x=x, keepdim=True)


@pytest.mark.api_base_sum_parameters
def test_sum9():
    """
    keepdim=True
    shape num > 65535
    """
    x = np.random.uniform(-1, 1, [66416, 20, 5])
    nsum = np.sum(x, axis=1, keepdims=True)
    paddle.disable_static()
    pdata = paddle.to_tensor(x)
    psum = paddle.sum(pdata, axis=1, keepdim=True)
    assert np.allclose(nsum[66000], psum[66000])


class TestSum1(APIBase):
    """
    test sum
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        self.enable_backward = False


obj1 = TestSum1(paddle.sum)


@pytest.mark.api_base_sum_parameters
def test_sum10():
    """
    input is int32, int64
    """
    x = np.array([[3, 5], [6, 2]])
    res = [np.sum(x)]
    obj1.run(res=res, x=x)
