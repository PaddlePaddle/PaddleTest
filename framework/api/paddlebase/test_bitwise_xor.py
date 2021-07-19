#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expxortab:ft=python
"""
test bitwise_xor
"""
# from apibase import APIBase
from apibase import compare

# from apibase import rxortool
import pytest
import paddle
import numpy as np


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor_base():
    """
    bitwise_xor
    base
    """
    x = np.array([-5, -1, 1]).astype(np.int32)
    y = np.array([4, 2, -3]).astype(np.int32)
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor1():
    """
    bitwise_xor
    input.type=bool
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.bool)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.bool)
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor2():
    """
    bitwise_xor
    input.type=uint8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.uint8)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.uint8) * 10
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor3():
    """
    bitwise_xor
    input.type=int8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int8)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int8) * 20
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor4():
    """
    bitwise_xor
    input.type=int16
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int16)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int16) * 23
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor5():
    """
    bitwise_xor
    input.type=int32
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int32)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int32) * 33
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_vartype
def test_tensor_bitwise_xor6():
    """
    bitwise_xor
    input.type=int64
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int64)
    y = np.ones((2, 3, 3, 3, 4, 1, 5, 2)).astype(np.int64) * 37
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_parameters
def test_tensor_bitwise_xor7():
    """
    bitwise_xor
    input.type=int64
    broadcast: x.shape=(2, 3, 1, 5)  y.shape=(3, 4, 1)
    """
    x = np.arange(-1, 2 * 3 * 5 - 1).reshape(2, 3, 1, 5).astype(np.int64)
    y = np.ones((3, 4, 1)).astype(np.int64) * 5
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_parameters
def test_tensor_bitwise_xor8():
    """
    bitwise_xor
    input.type=int64
    broadcast: x.shape=(3, 4, 1)  y.shape=(2, 3, 1, 5)
    """
    x = np.arange(-1, 3 * 4 - 1).reshape(3, 4, 1).astype(np.int64)
    y = np.ones((2, 3, 1, 5)).astype(np.int64) * 3
    exp = np.bitwise_xor(x, y)
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y))
    compare(exp, res.numpy())
    res1 = paddle.to_tensor(x) ^ paddle.to_tensor(y)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_xor_parameters
def test_tensor_bitwise_xor9():
    """
    bitwise_xor
    input.type=int64
    broadcast: x.shape=(3, 4, 1)  y.shape=(2, 3, 1, 5)
    """
    x = np.arange(-1, 3 * 4 - 1).reshape(3, 4, 1).astype(np.int64)
    y = np.ones((2, 3, 1, 5)).astype(np.int64) * 3
    exp = np.bitwise_xor(x, y)
    tmp = paddle.to_tensor(np.ones((2, 3, 4, 5)).astype(np.int64))
    res = paddle.bitwise_xor(paddle.to_tensor(x), paddle.to_tensor(y), out=tmp)
    compare(exp, res.numpy())
