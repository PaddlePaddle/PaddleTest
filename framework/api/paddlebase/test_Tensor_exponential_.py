#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_exponential_
"""

import numpy as np
import paddle
import pytest


@pytest.mark.api_base_exponential_parameters
def test_exponential_0():
    """
    test01
    """
    paddle.seed(33)
    x = np.array([0.4, 0.5])
    xp = paddle.to_tensor(x)
    if paddle.is_compiled_with_cuda():
        return
    res = paddle.Tensor.exponential_(xp, 1).numpy()
    res_np = np.array([2.92539735, 5.97663453])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_exponential_parameters
def test_exponential_1():
    """
    test02
    """
    paddle.seed(33)
    x = np.array([0.4, 0.5])
    xp = paddle.to_tensor(x)
    if not paddle.is_compiled_with_cuda():
        return
    res = paddle.Tensor.exponential_(xp, 1).numpy()
    res_np = np.array([3.28828073, 1.04046203])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_exponential_parameters
def test_exponential_2():
    """
    test03
    """
    paddle.seed(33)
    x = np.ones((2, 2))
    xp = paddle.to_tensor(x)
    if paddle.is_compiled_with_cuda():
        return
    res = paddle.Tensor.exponential_(xp, 1).numpy()
    res_np = np.array([[2.92539735, 5.97663453], [0.86001479, 2.24217270]])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_exponential_parameters
def test_exponential_3():
    """
    test04
    """
    paddle.seed(33)
    x = np.ones((2, 2))
    xp = paddle.to_tensor(x)
    if not paddle.is_compiled_with_cuda():
        return
    res = paddle.Tensor.exponential_(xp, 1).numpy()
    res_np = np.array([[3.28828073, 1.04046203], [0.40575318, 0.74771885]])
    assert np.allclose(res, res_np)
