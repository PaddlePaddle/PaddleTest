#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test einsum
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


x = randtool("float", -10, 10, [10])
y = randtool("float", -10, 10, [6])
x1 = randtool("float", -10, 10, [3, 10, 3])
y1 = randtool("float", -10, 10, [3, 3, 10])


@pytest.mark.api_base_einsum_parameters
def test_einsum():
    """
    test sum op = 'i->'
    """
    op = "i->"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x))
    expect = np.einsum(op, x)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum1():
    """
    test dot op = 'i,i->'
    """
    op = "i,i->"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x), paddle.to_tensor(x))
    expect = np.einsum(op, x, x)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum2():
    """
    test outer op = 'i,j->ij'
    """
    op = "i,j->ij"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x), paddle.to_tensor(y))
    expect = np.einsum(op, x, y)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum3():
    """
    test batch matrix multiplication op = 'ijk, ikl->ijl'
    """
    op = "ijk, ikl->ijl"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x1), paddle.to_tensor(y1))
    expect = np.einsum(op, x1, y1)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum4():
    """
    test transpose op = 'ijk->kji'
    """
    op = "ijk->kji"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x1))
    expect = np.einsum(op, x1)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum5():
    """
    test  Ellipsis transpose op = '...jk->...kj'
    """
    op = "...jk->...kj"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x1))
    expect = np.einsum(op, x1)
    np.allclose(paddle_res.numpy(), expect)


@pytest.mark.api_base_einsum_parameters
def test_einsum6():
    """
    test batch matrix multiplication op = '...jk, ...kl->...jl'
    """
    op = "...jk, ...kl->...jl"
    paddle_res = paddle.einsum(op, paddle.to_tensor(x1), paddle.to_tensor(y1))
    expect = np.einsum(op, x1, y1)
    np.allclose(paddle_res.numpy(), expect)
