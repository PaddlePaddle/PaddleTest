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


@pytest.mark.api_base_where_parameters
def test_where4():
    """
    condition.shape=[2, 4]
    a.shape=[2, 2, 4]
    b.shape=[2, 2, 4]
    """
    np.random.seed(103)
    x = np.random.random((2, 2, 4)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.random.random((2, 2, 4)).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    in_tensor = paddle.to_tensor(np.random.random((2, 4)))
    out_tensor = paddle.where(in_tensor > 0.5, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array(
        [
            [[0.4321112, 0.35727233, 0.48606, 0.8782824], [0.7549087, 0.45935374, 0.82268476, 0.5901616]],
            [[0.30712026, 0.59614646, 0.09247963, 0.40227097], [0.5896865, 0.6089751, 0.67236334, 0.05976145]],
        ]
    ).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where5():
    """
    condition.shape=[2, 1, 4]
    a.shape=[2, 1, 4]
    b.shape=[2, 2, 4]
    """
    np.random.seed(103)
    x = np.random.random((2, 1, 4)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.random.random((2, 2, 4)).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    in_tensor = paddle.to_tensor(np.random.random((2, 1, 4)))
    out_tensor = paddle.where(in_tensor > 0.5, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array(
        [
            [[0.4321112, 0.17421526, 0.40323246, 0.9478893], [0.4321112, 0.17421526, 0.67236334, 0.00798261]],
            [[0.58717126, 0.35727233, 0.48606, 0.8782824], [0.58717126, 0.630465, 0.37648368, 0.5901616]],
        ]
    ).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where6():
    """
    condition.shape=[3, 2, 2, 4]
    a.shape=[2, 2, 4]
    b.shape=[2, 2, 4]
    """
    np.random.seed(103)
    x = np.random.random((2, 2, 4)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.random.random((2, 2, 4)).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    in_tensor = paddle.to_tensor(np.random.random((3, 2, 2, 4)))
    out_tensor = paddle.where(in_tensor > 0.5, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array(
        [
            [
                [[0.4321112, 0.35727233, 0.48606, 0.8782824], [0.7549087, 0.45935374, 0.82268476, 0.5901616]],
                [[0.30712026, 0.59614646, 0.40323246, 0.9478893], [0.6775444, 0.6089751, 0.67236334, 0.05976145]],
            ],
            [
                [[0.4321112, 0.17421526, 0.48606, 0.8782824], [0.7549087, 0.630465, 0.37648368, 0.5901616]],
                [[0.30712026, 0.59614646, 0.40323246, 0.40227097], [0.6775444, 0.11541288, 0.67236334, 0.05976145]],
            ],
            [
                [[0.33655992, 0.17421526, 0.48606, 0.8782824], [0.7549087, 0.630465, 0.82268476, 0.8215481]],
                [[0.73732877, 0.20089437, 0.09247963, 0.40227097], [0.6775444, 0.6089751, 0.67236334, 0.00798261]],
            ],
        ]
    ).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where7():
    """
    condition.shape=[2, 1, 4]
    a.shape=[2, 2, 1]
    b.shape=[2, 2, 1]
    """
    np.random.seed(103)
    x = np.random.random((2, 2, 1)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.random.random((2, 2, 1)).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    in_tensor = paddle.to_tensor(np.random.random((2, 1, 4)))
    out_tensor = paddle.where(in_tensor > 0.5, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array(
        [
            [[0.58717126, 0.58717126, 0.58717126, 0.4321112], [0.45935374, 0.45935374, 0.45935374, 0.17421526]],
            [[0.17094369, 0.17094369, 0.17094369, 0.82268476], [0.82763225, 0.82763225, 0.82763225, 0.8215481]],
        ]
    ).astype(np.float32)
    compare(out, res)


@pytest.mark.api_base_where_parameters
def test_where8():
    """
    condition.shape=[2, 1, 4]
    a.shape=[2, 1, 1]
    b.shape=[2, 2, 4]
    """
    np.random.seed(103)
    x = np.random.random((2, 1, 1)).astype(np.float32)
    x_tensor = paddle.to_tensor(x)
    y = np.random.random((2, 2, 4)).astype(np.float32)
    y_tensor = paddle.to_tensor(y)
    in_tensor = paddle.to_tensor(np.random.random((2, 1, 4)))
    out_tensor = paddle.where(in_tensor > 0.5, x_tensor, y_tensor)
    out = out_tensor.numpy()
    res = np.array(
        [
            [[0.17094369, 0.4321112, 0.4321112, 0.4321112], [0.82268476, 0.4321112, 0.4321112, 0.4321112]],
            [[0.40323246, 0.17421526, 0.17421526, 0.17421526], [0.67236334, 0.17421526, 0.17421526, 0.17421526]],
        ]
    ).astype(np.float32)
    compare(out, res)
