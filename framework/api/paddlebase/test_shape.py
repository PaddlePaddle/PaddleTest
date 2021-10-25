#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test shape
"""
from apibase import APIBase
from apibase import compare
import paddle.fluid as fluid

import paddle
import pytest
import numpy as np


if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]


@pytest.mark.api_base_shape_vartype
def test_shape_base():
    """
    base
    """
    types = [np.float16, np.float32, np.float64, np.int32, np.int64]
    for t in types:
        x = np.random.random((2, 3, 5, 5)).astype(t)
        for place in places:
            paddle.disable_static(place)
            res = paddle.shape(paddle.to_tensor(x))
            exp = np.array(x.shape)
            compare(exp, res.numpy())
            paddle.enable_static()


@pytest.mark.api_base_shape_parameters
def test_shape_base():
    """
    base
    """
    x = np.array([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())


@pytest.mark.api_base_shape_parameters
def test_shape1():
    """
    x = np.nan
    """
    x = np.array([np.nan, np.inf, np.NINF, -np.inf, -np.nan])
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())


@pytest.mark.api_base_shape_parameters
def test_shape2():
    """
    x =np.zeros([2])
    """
    x = np.zeros([2])
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())


@pytest.mark.api_base_shape_parameters
def test_shape4():
    """
    x =np.ones([2])
    """
    x = np.ones([2])
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())


@pytest.mark.api_base_shape_parameters
def test_shape5():
    """
    x =np.random.random((2, 3, 5, 5, 2, 1, 2, 2, 2))
    """
    x = np.random.random((2, 3, 5, 5, 2, 1, 2, 2, 2))
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())


@pytest.mark.api_base_shape_parameters
def test_shape6():
    """
    x =np.array([[]] * 3)
    """
    x = np.array([[]] * 3)
    res = paddle.shape(paddle.to_tensor(x))
    exp = np.array(x.shape)
    compare(exp, res.numpy())
