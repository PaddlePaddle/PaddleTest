#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test where
"""
import paddle
import numpy as np
import numpy.testing as npt
from apibase import compare
import pytest


if paddle.device.is_compiled_with_cuda():
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]

types = [np.float32, np.float64, np.int32, np.int64]


@pytest.mark.api_base_where_vartype
def test_where():
    """
    condition: x > y
    """
    for t in types:
        x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype(t)
        x_tensor = paddle.to_tensor(x)
        y = np.array([0.9421, 0.0233, 2.1, 4.3]).astype(t)
        y_tensor = paddle.to_tensor(y)
        for d in devices:
            paddle.set_device(d)
            out_tensor = paddle.where(x_tensor > y_tensor, x_tensor, y_tensor)
            out = out_tensor.numpy()
            res = np.array([0.9421, 0.1983, 3.2, 4.3]).astype(t)
            compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where1():
    """
    condition: x > 2
    """
    x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.array([0.9421, 0.0233, 2.1, 4.3]).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    out_tensor = paddle.where(x_tensor > 2, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array([0.9421, 0.0233, 3.2, 4.3]).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where2():
    """
    condition: [3, 3, 3, 3] > -1
    """
    x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.array([0.9421, 0.0233, 2.1, 4.3]).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    out_tensor = paddle.where(paddle.to_tensor([3, 3, 3, 3]) > -1, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array([0.9383, 0.1983, 3.2, 1.2]).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where3():
    """
    condition: -1 > [2, 2, 2, 2]
    """
    x = np.array([0.9383, 0.1983, 3.2, 1.2]).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.array([0.9421, 0.0233, 2.1, 4.3]).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    out_tensor = paddle.where(-1 > paddle.to_tensor([2, 2, 2, 2]), x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array([0.9421, 0.0233, 2.1, 4.3]).astype(np.float32)
    compare(out, res)
