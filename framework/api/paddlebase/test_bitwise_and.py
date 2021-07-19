#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test bitwise_and
"""
# from apibase import APIBase
from apibase import compare

# from apibase import randtool
import pytest
import paddle
import numpy as np


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and_base():
    """
    bitwise_and
    base
    """
    x = np.array([-5, -1, 1]).astype(np.int32)
    y = np.array([4, 2, -3]).astype(np.int32)
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and1():
    """
    bitwise_and
    input.type=bool
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.bool)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.bool)
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and2():
    """
    bitwise_and
    input.type=uint8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.uint8)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.uint8) * 10
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and3():
    """
    bitwise_and
    input.type=int8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int8)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int8) * 20
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and4():
    """
    bitwise_and
    input.type=int16
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int16)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int16) * 23
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and5():
    """
    bitwise_and
    input.type=int32
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int32)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int32) * 33
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_vartype
def test_tensor_bitwise_and6():
    """
    bitwise_and
    input.type=int64
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int64)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int64) * 37
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_parameters
def test_tensor_bitwise_and7():
    """
    bitwise_and
    input.type=int64
    broadcast: x.shape=(2, 3, 1, 5)  y.shape=(3, 4, 1)
    """
    x = np.arange(-1, 2 * 3 * 5 - 1).reshape(2, 3, 1, 5).astype(np.int64)
    y = np.ones((3, 4, 1)).astype(np.int64) * 5
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_parameters
def test_tensor_bitwise_and8():
    """
    bitwise_and
    input.type=int64
    broadcast: x.shape=(3, 4, 1)  y.shape=(2, 3, 1, 5)
    """
    x = np.arange(-1, 3 * 4 - 1).reshape(3, 4, 1).astype(np.int64)
    y = np.ones((2, 3, 1, 5)).astype(np.int64) * 3
    exp = np.bitwise_and(x, y)
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) & paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_and_parameters
def test_tensor_bitwise_and9():
    """
    bitwise_and
    input.type=int64
    broadcast: x.shape=(3, 4, 1)  y.shape=(2, 3, 1, 5)
    """
    x = np.arange(-1, 3 * 4 - 1).reshape(3, 4, 1).astype(np.int64)
    y = np.ones((2, 3, 1, 5)).astype(np.int64) * 3
    exp = np.bitwise_and(x, y)
    tmp = paddle.to_tensor(np.ones((2, 3, 4, 5)).astype(np.int64))
    res = paddle.bitwise_and(paddle.to_tensor(x), paddle.to_tensor(y), out=tmp)
    compare(exp, res.numpy())
