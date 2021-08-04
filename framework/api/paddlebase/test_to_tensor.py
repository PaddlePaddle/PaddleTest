#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test to_tensor
"""
from apibase import compare
import paddle
import numpy as np
import pytest


if paddle.device.is_compiled_with_cuda():
    devices = ["gpu", "cpu"]
else:
    devices = ["cpu"]


tensor_types = [
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "complex64",
    "complex128",
]


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor():
    """
    x is np.array
    """
    for t in tensor_types:
        exp = np.arange(1, 7).reshape((2, 3)).astype(t)
        for d in devices:
            paddle.set_device(d)
            for s in [True, False]:
                res = paddle.to_tensor(exp, stop_gradient=s)
                compare(exp, res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor1():
    """
    x is list
    """
    exp = [[1.1, 2.3, 4.2], [1.1, 2.3, 4.2], [1.1, 2.3, 4.2], [1.1, 2.3, 4.2]]
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor2():
    """
    x is tuple
    """
    exp = ((1.1, 2.3, 4.2), (1.1, 2.3, 4.2), (1.1, 2.3, 4.2), (1.1, 2.3, 4.2))
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s


@pytest.mark.api_base_to_tensor_parameters
def test_to_tensor3():
    """
    x is complex list
    """
    exp = [[1.1 + 1j, 2.3, 4.2], [1.1, 2.3 - 5j, 4.2], [1.1, 2.3 + 2j, 4.2], [1.1, 2.3, 4.2]]
    for d in devices:
        paddle.set_device(d)
        for t in tensor_types:
            for s in [True, False]:
                res = paddle.to_tensor(exp, dtype=t, stop_gradient=s)
                compare(np.array(exp).astype(t), res, delta=1e-20, rtol=1e-20)
                assert res.stop_gradient is s
