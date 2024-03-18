#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_poisson
"""

import numpy as np
import paddle
import pytest

paddle.seed(33)


@pytest.mark.api_base_poisson_parameters
def test_poisson0():
    """
    test01
    """
    x = np.array([0.4, 0.5])
    xp = paddle.to_tensor(x)
    if paddle.is_compiled_with_cuda():
        return
    res = paddle.poisson(xp).numpy()
    res_np = np.array([2.0, 1.0])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_poisson_parameters
def test_poisson1():
    """
    test02
    """
    x = np.array([0.4, 0.5])
    xp = paddle.to_tensor(x)
    if not paddle.is_compiled_with_cuda():
        return
    res = paddle.poisson(xp).numpy()
    res_np = np.array([1.0, 1.0])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_poisson_parameters
def test_poisson2():
    """
    test03
    """
    x = np.array([[0.4, 0.1], [0.5, 0.2]])
    xp = paddle.to_tensor(x)
    if paddle.is_compiled_with_cuda():
        return
    res = paddle.poisson(xp).numpy()
    res_np = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(res, res_np)


@pytest.mark.api_base_poisson_parameters
def test_poisson3():
    """
    test04
    """
    x = np.array([[0.4, 0.1], [0.5, 0.2]])
    xp = paddle.to_tensor(x)
    if not paddle.is_compiled_with_cuda():
        return
    res = paddle.poisson(xp).numpy()
    res_np = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert np.allclose(res, res_np)
