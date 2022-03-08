#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Orthogonal
"""

import numpy as np
import paddle
import pytest


@pytest.mark.api_nn_Orthogonal_parameters
def test_orthogonal0():
    """
    linear
    """
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr)
    x = linear.weight.numpy()
    orth = np.dot(x, x.T)
    res = np.diag([1.0] * 2)
    assert np.allclose(res, orth, atol=1e-7)


@pytest.mark.api_nn_Orthogonal_parameters
def test_orthogonal1():
    """
    linear
    """
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
    linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
    x = linear.weight.numpy()
    orth = np.dot(x, x.T)
    res = np.diag([1.0] * 10)
    assert np.allclose(res, orth, atol=1e-6)


@pytest.mark.api_nn_Orthogonal_parameters
def test_orthogonal2():
    """
    linear
    """
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
    linear = paddle.nn.Linear(15, 10, weight_attr=weight_attr)
    x = linear.weight.numpy()
    orth = np.dot(x.T, x)
    res = np.diag([1.0] * 10)
    assert np.allclose(res, orth, atol=1e-6)


@pytest.mark.api_nn_Orthogonal_parameters
def test_orthogonal3():
    """
    conv1d
    """
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
    conv = paddle.nn.Conv1D(2, 3, 4, weight_attr=weight_attr)
    x = conv.weight.numpy().reshape(3, 8)
    orth = np.dot(x, x.T)
    res = np.diag([1.0] * 3)
    assert np.allclose(res, orth, atol=1e-6)
