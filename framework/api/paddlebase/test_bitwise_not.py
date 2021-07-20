#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
test bitwise_not
"""
# from apibase import APIBase
from apibase import compare

# from apibase import rnottool
import pytest
import paddle
import numpy as np


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not_base():
    """
    bitwise_not
    base
    """
    x = np.array([-5, -1, 1]).astype(np.int32)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not1():
    """
    bitwise_not
    input.type=bool
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.bool)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not2():
    """
    bitwise_not
    input.type=uint8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.uint8)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not3():
    """
    bitwise_not
    input.type=int8
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int8)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not4():
    """
    bitwise_not
    input.type=int16
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int16)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not5():
    """
    bitwise_not
    input.type=int32
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int32)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_vartype
def test_tensor_bitwise_not6():
    """
    bitwise_not
    input.type=int64
    """
    x = np.arange(-1, 2 * 3 * 3 * 3 * 4 * 1 * 5 * 2 - 1).reshape(2, 3, 3, 3, 4, 1, 5, 2).astype(np.int64)
    exp = np.bitwise_not(x)
    res = paddle.bitwise_not(paddle.to_tensor(x))
    compare(exp, res.numpy())
    res1 = ~paddle.to_tensor(x)
    compare(exp, res1.numpy())


@pytest.mark.api_base_bitwise_not_parameters
def test_tensor_bitwise_not7():
    """
    bitwise_not
    input.type=int64
    broadcast: x.shape=(3, 4, 1)  y.shape=(2, 3, 1, 5)
    """
    x = np.arange(-1, 3 * 4 - 1).reshape(3, 4, 1).astype(np.int64)
    exp = np.bitwise_not(x)
    tmp = paddle.to_tensor(np.ones((3, 4, 1)).astype(np.int64))
    tmp = paddle.bitwise_not(paddle.to_tensor(x), out=tmp)
    compare(exp, tmp.numpy())
