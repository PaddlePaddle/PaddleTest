#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_add_op_cpu_jit
"""
import os
import paddle
from paddle.utils.cpp_extension import load
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))


def test_add_op_jit():
    """
    test add op jit
    Returns:

    """
    paddle.set_device("cpu")
    paddle.seed(33)
    custom_ops = load(name="add_op_jit", sources=[current_path + "/add_op.cc"])

    x = paddle.to_tensor(np.array([[1, 1], [1, 1]]).astype("float32"))
    x1 = paddle.to_tensor(np.array([[1, 1], [1, 1]]).astype("float32"))
    x.stop_gradient = False
    x1.stop_gradient = False
    print(x)
    out = custom_ops.add_test(x, x1)
    assert np.allclose(out.numpy(), np.array([[2, 2], [2, 2]]).astype("float32"))
    out.backward()
    assert np.allclose(out.grad, np.array([[1, 1], [1, 1]]).astype("float32"))
    assert np.allclose(x.grad, np.array([[1, 1], [1, 1]]).astype("float32"))
    assert np.allclose(x1.grad, np.array([[1, 1], [1, 1]]).astype("float32"))
