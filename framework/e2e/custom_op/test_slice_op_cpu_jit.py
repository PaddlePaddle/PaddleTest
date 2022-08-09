#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
slice op
"""
import os
import paddle
from paddle.utils.cpp_extension import load
import numpy as np
import pytest

current_path = os.path.dirname(os.path.abspath(__file__))


paddle.set_device("cpu")
paddle.seed(33)
custom_ops = load(name="slice_op_jit", sources=[current_path + "/slice_op.cc"])


def test_add_op_jit():
    """
    test slice op jit
    Returns:

    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    a = 1
    b = 5
    out = custom_ops.slice_test(paddle_x, a, b)
    assert np.allclose(out.numpy(), x[a:b])


def test_add_op_jit1():
    """
    test slice op  a=start b=end
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    a = 0
    b = 6
    out = custom_ops.slice_test(paddle_x, a, b)
    assert np.allclose(out.numpy(), x[a:b])


def test_add_op_jit2():
    """
    test slice op exception a=b
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    a = 4
    b = 4
    etype = ValueError
    with pytest.raises(etype):
        custom_ops.slice_test(paddle_x, a, b)


def test_add_op_jit3():
    """
    test slice op exception a>b
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    a = 5
    b = 4
    etype = ValueError
    with pytest.raises(etype):
        custom_ops.slice_test(paddle_x, a, b)


def test_add_op_jit4():
    """
    test slice op exception out of bound
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    paddle_x = paddle.to_tensor(x).astype("float32")
    paddle_x.stop_gradient = False
    a = 5
    b = 33
    etype = IndexError
    with pytest.raises(etype):
        custom_ops.slice_test(paddle_x, a, b)
