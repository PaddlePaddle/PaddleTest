#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_einsum
"""

import paddle
import pytest
import numpy as np


@pytest.mark.api_base_einsum_parameters
def test_einsum0():
    """
    sum
    """
    x = np.random.rand(4, 5)
    xp = paddle.to_tensor(x)
    api_res = paddle.einsum("ij->", xp).numpy()
    res = np.sum(x)
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum1():
    """
    1d-tensor
    dot
    """
    x = np.random.rand(4)
    xp = paddle.to_tensor(x)
    api_res = paddle.einsum("i, i->", xp, xp).numpy()
    res = np.dot(x, x)
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum2():
    """
    1d-tensor
    outer
    """
    x = np.random.rand(4)
    y = np.random.rand(5)
    xp = paddle.to_tensor(x)
    yp = paddle.to_tensor(y)
    api_res = paddle.einsum("i, j->ij", xp, yp).numpy()
    res = np.outer(x, y)
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum3():
    """
    transpose
    """
    x = np.random.rand(4, 5, 6)
    xp = paddle.to_tensor(x)
    api_res = paddle.einsum("ijk->ikj", xp).numpy()
    res = np.transpose(x, (0, 2, 1))
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum4():
    """
    multiplication
    """
    x = np.random.rand(4, 5, 6)
    y = np.random.rand(4, 6, 8)
    xp, yp = paddle.to_tensor(x), paddle.to_tensor(y)
    api_res = paddle.einsum("ijk, ikl->ijl", xp, yp).numpy()
    res = np.matmul(x, y)
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum5():
    """
    multiplication
    """
    x = np.random.rand(4, 5, 6)
    y = np.random.rand(4, 6, 8)
    xp, yp = paddle.to_tensor(x), paddle.to_tensor(y)
    api_res = paddle.einsum("ijk, ikl->ijl", xp, yp).numpy()
    res = np.matmul(x, y)
    assert np.allclose(api_res, res)


@pytest.mark.api_base_einsum_parameters
def test_einsum6():
    """
    multiplication
    """
    x = np.random.rand(4, 5, 6, 7, 8)
    y = np.random.rand(4, 5, 6, 8, 2)
    xp, yp = paddle.to_tensor(x), paddle.to_tensor(y)
    api_res = paddle.einsum("...jk, ...kl->...jl", xp, yp).numpy()
    res = np.matmul(x, y)
    assert np.allclose(api_res, res)
