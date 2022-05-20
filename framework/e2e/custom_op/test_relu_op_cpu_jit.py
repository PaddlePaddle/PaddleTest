#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_relu_op_cpu_jit
"""
import os
import paddle
from paddle.utils.cpp_extension import load
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))


def test_relu_op_jit():
    """
    test relu op jit
    Returns:

    """
    paddle.set_device("cpu")
    paddle.seed(33)
    np.random.seed(33)
    custom_ops = load(name="relu_op_jit", sources=[current_path + "/relu_op.cc"])
    data = np.array([[[0.3, 0.7], [-1, -0.4]], [[0.2, 0.4], [-0.2, -0.5]]]).astype("float32")
    x = paddle.to_tensor(data)
    x.stop_gradient = False
    out = custom_ops.custom_relu(x)
    print(out.numpy())
    print(np.maximum(data, 0))
    assert np.allclose(out.numpy(), np.maximum(data, 0).astype("float32"))
    out.backward()
    assert np.allclose(out.grad, np.ones(shape=[2, 2, 2]).astype("float32"))
