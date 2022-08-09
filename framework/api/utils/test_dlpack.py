#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test to_dlpack and from_dlpack
"""

import paddle
import pytest
import numpy as np


@pytest.mark.api_utils_dlpack_base
def test_dlpack_base():
    """
    x: 1d-tensor
    """
    xp = np.random.randn(3, 4)
    types = [
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
    for dtype in types:
        x1 = paddle.to_tensor(xp, dtype=dtype)
        dlpack = paddle.utils.dlpack.to_dlpack(x1)
        x2 = paddle.utils.dlpack.from_dlpack(dlpack)
        assert np.allclose(x1.numpy(), x2.numpy())


@pytest.mark.api_utils_dlpack_parameters
def test_dlpack0():
    """
    x: 1d-tensor
    """
    xp = np.random.randn(3)
    x1 = paddle.to_tensor(xp)
    dlpack = paddle.utils.dlpack.to_dlpack(x1)
    x2 = paddle.utils.dlpack.from_dlpack(dlpack)
    assert np.allclose(x1.numpy(), x2.numpy())


@pytest.mark.api_utils_dlpack_parameters
def test_dlpack1():
    """
    x: > 2d
    """
    xp = np.random.randn(3, 4, 5, 1, 2, 2, 3, 4, 6)
    x1 = paddle.to_tensor(xp)
    dlpack = paddle.utils.dlpack.to_dlpack(x1)
    x2 = paddle.utils.dlpack.from_dlpack(dlpack)
    assert np.allclose(x1.numpy(), x2.numpy())
