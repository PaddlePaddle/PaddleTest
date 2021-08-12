#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test broadcast_tensors
"""
import math

from paddle.fluid.data import data
from paddle.fluid.layers.nn import pad

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


@pytest.mark.api_base_broadcast_tensors_vartype
def test_broadcast_tensors_base():
    """
    base 2-D tensor
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [6, 6]).astype(i)
        y = randtool("float", -5, 5, [6, 1]).astype(i)
        res = np.broadcast_arrays(x, y)
        paddle_res = paddle.broadcast_tensors(input=[paddle.to_tensor(x), paddle.to_tensor(y)])
        paddle_res_numpy = []
        for i in paddle_res:
            paddle_res_numpy.append(i.numpy())
        assert np.allclose(res, paddle_res_numpy)


@pytest.mark.api_base_broadcast_tensors_vartype
def test_broadcast_tensors_base1():
    """
    base 3-D tensor
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [6, 6, 6]).astype(i)
        y = randtool("float", -5, 5, [6, 1, 6]).astype(i)
        res = np.broadcast_arrays(x, y)
        paddle_res = paddle.broadcast_tensors(input=[paddle.to_tensor(x), paddle.to_tensor(y)])
        paddle_res_numpy = []
        for i in paddle_res:
            paddle_res_numpy.append(i.numpy())
        assert np.allclose(res, paddle_res_numpy)


@pytest.mark.api_base_broadcast_tensors_vartype
def test_broadcast_tensors_base2():
    """
    base 4-D tensor
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [6, 6, 6, 1]).astype(i)
        y = randtool("float", -5, 5, [6, 1, 6, 3]).astype(i)
        res = np.broadcast_arrays(x, y)
        paddle_res = paddle.broadcast_tensors(input=[paddle.to_tensor(x), paddle.to_tensor(y)])
        paddle_res_numpy = []
        for i in paddle_res:
            paddle_res_numpy.append(i.numpy())
        assert np.allclose(res, paddle_res_numpy)


@pytest.mark.api_base_broadcast_tensors_vartype
def test_broadcast_tensors_base3():
    """
    base 5-D tensor
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [1, 6, 6, 6, 1]).astype(i)
        y = randtool("float", -5, 5, [1, 6, 1, 6, 3]).astype(i)
        res = np.broadcast_arrays(x, y)
        paddle_res = paddle.broadcast_tensors(input=[paddle.to_tensor(x), paddle.to_tensor(y)])
        paddle_res_numpy = []
        for i in paddle_res:
            paddle_res_numpy.append(i.numpy())
        assert np.allclose(res, paddle_res_numpy)


@pytest.mark.api_base_broadcast_tensors_parameters
def test_broadcast_tensors():
    """
    base 4 tensors to broadcast
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [1, 6, 2, 6, 1]).astype(i)
        y = randtool("float", -5, 5, [1, 6, 1, 1, 3]).astype(i)
        z = randtool("float", -5, 5, [1, 1, 1, 6, 3]).astype(i)
        s = randtool("float", -5, 5, [1, 6, 1, 6, 3]).astype(i)

        res = np.broadcast_arrays(x, y, z, s)
        paddle_res = paddle.broadcast_tensors(
            input=[paddle.to_tensor(x), paddle.to_tensor(y), paddle.to_tensor(z), paddle.to_tensor(s)]
        )
        paddle_res_numpy = []
        for i in paddle_res:
            paddle_res_numpy.append(i.numpy())
        assert np.allclose(res, paddle_res_numpy)


@pytest.mark.api_base_broadcast_tensors_exception
def test_broadcast_tensors1():
    """
    base 4 tensors to broadcast
    """
    datatype = [np.bool8, np.int32, np.int64, np.float32, np.float64]
    for i in datatype:
        x = randtool("float", -5, 5, [1, 6, 2, 6, 1]).astype(i)
        y = randtool("float", -5, 5, [1, 6, 1, 1, 3]).astype(i)
        z = randtool("float", -5, 5, [1, 1, 1, 2, 3]).astype(i)
        s = randtool("float", -5, 5, [1, 6, 1, 6, 3]).astype(i)
        # etype = ValueError
        # with pytest.raises(etype):
        paddle.broadcast_tensors(
            input=[paddle.to_tensor(x), paddle.to_tensor(y), paddle.to_tensor(z), paddle.to_tensor(s)]
        )
