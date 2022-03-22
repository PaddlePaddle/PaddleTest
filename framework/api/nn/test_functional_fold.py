#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test functional_fold
"""

import paddle
import pytest
import numpy as np


@pytest.mark.api_nn_functional_fold_base
def test_functional_fold_base():
    """
    test base
    """
    xp = np.random.randn(3, 3 * 2 * 2, 12)
    types = ["float32", "float64"]
    for dtype in types:
        x = paddle.to_tensor(xp, dtype=dtype)
        y = paddle.nn.functional.fold(x, output_sizes=[4, 5], kernel_sizes=2)
        assert y.shape == [3, 3, 4, 5]


@pytest.mark.api_nn_functional_fold_parameters
def test_functional_fold0():
    """
    output_size: int
    """
    xp = np.random.randn(3, 3 * 2 * 2, 9)
    x = paddle.to_tensor(xp)
    y = paddle.nn.functional.fold(x, output_sizes=4, kernel_sizes=2)
    assert y.shape == [3, 3, 4, 4]


@pytest.mark.api_nn_functional_fold_parameters
def test_functional_fold1():
    """
    kernel_size: tuple
    """
    xp = np.random.randn(2, 12, 6)
    x = paddle.to_tensor(xp)
    y = paddle.nn.functional.fold(x, output_sizes=4, kernel_sizes=(2, 3))
    assert y.shape == [2, 2, 4, 4]


@pytest.mark.api_nn_functional_fold_parameters
def test_functional_fold2():
    """
    kernel_size: tuple
    stride: 2
    """
    xp = np.random.randn(2, 12, 2)
    x = paddle.to_tensor(xp)
    y = paddle.nn.functional.fold(x, output_sizes=4, kernel_sizes=(2, 3), strides=2)
    assert y.shape == [2, 2, 4, 4]


@pytest.mark.api_nn_functional_fold_parameters
def test_functional_fold3():
    """
    kernel_size: tuple
    stride: 2
    paddings = 1
    """
    xp = np.random.randn(2, 12, 6)
    x = paddle.to_tensor(xp)
    y = paddle.nn.functional.fold(x, output_sizes=4, kernel_sizes=(2, 3), strides=2, paddings=1)
    assert y.shape == [2, 2, 4, 4]


@pytest.mark.api_nn_functional_fold_parameters
def test_functional_fold4():
    """
    kernel_size: tuple
    stride: 2
    paddings = 1
    dilations = 2
    """
    xp = np.random.randn(2, 12, 2)
    x = paddle.to_tensor(xp)
    y = paddle.nn.functional.fold(x, output_sizes=4, kernel_sizes=(2, 3), strides=2, paddings=1, dilations=2)
    assert y.shape == [2, 2, 4, 4]
