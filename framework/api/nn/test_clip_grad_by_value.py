#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clip_grad_by_value
"""

from apibase import APIBase
from apibase import randtool, compare
import paddle
import pytest
import numpy as np


def numpy_clip_grad_by_value(test_data, clip_max, clip_min=None):
    """
    ClipGradByValue implemented by numpy.
    """
    cliped_data = []
    for data, grad in test_data:
        grad = np.clip(grad, clip_min, clip_max)
        cliped_data.append((data, grad))
    return cliped_data


def generate_test_data(length, shape, dtype="float32"):
    """
    generate test data
    """
    tensor_data = []
    numpy_data = []
    np.random.seed(100)
    for i in range(length):
        np_weight = np.random.rand(*shape)
        np_weight_grad = np.random.rand(*shape)
        numpy_data.append((np_weight, np_weight_grad))

        tensor_weight = paddle.to_tensor(np_weight)
        tensor_weight_grad = paddle.to_tensor(np_weight_grad)
        tensor_data.append((tensor_weight, tensor_weight_grad))
    return numpy_data, tensor_data


@pytest.mark.api_nn_ClipGradByValue_vartype
def test_clip_grad_by_value_base():
    """
    Test base.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = None
    clip_max = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm1():
    """
    Test ClipGradByValue when input shape changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        input grad shape: [10, 10] -> [4, 10, 10]
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [4, 10, 10]
    length = 4
    clip_min = None
    clip_max = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm2():
    """
    Test ClipGradByValue when input data dtype changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        input data dtype: float32 -> float64
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = None
    clip_max = 1.0
    dtype = "float64"
    np_data, paddle_data = generate_test_data(length, shape, dtype=dtype)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm3():
    """
   Test ClipGradByValue when clip_min and clip_max changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        input parameters: clip_min = None -> clip_min = -2.0, clip_max = 1.0 -> clip_max = -1.0
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = -2.0
    clip_max = -1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm4():
    """
   Test ClipGradByValue when clip_min and clip_max are equal.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        input parameters: clip_min = None -> clip_min = 2.0, clip_max = 1.0 -> clip_max = 2.0
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = 2.0
    clip_max = 2.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm5():
    """
   Test ClipGradByValue when input type changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        input data dtype: float32 -> float16
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    dtype = "float16"
    clip_min = None
    clip_max = 1.0
    np_data, paddle_data = generate_test_data(length, shape, dtype=dtype)
    np_res = numpy_clip_grad_by_value(np_data, clip_max=clip_max, clip_min=clip_min)

    paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm6():
    """
   Test ClipGradByValue when clip_max value changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        clip_max: 1.0 -> -1.0
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = None
    clip_max = -1.0
    np_data, paddle_data = generate_test_data(length, shape)

    with pytest.raises(AssertionError):
        paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
        paddle_clip(paddle_data)


@pytest.mark.api_nn_ClipGradByValue_parameters
def test_clip_grad_by_value_norm7():
    """
   Test ClipGradByValue when clip_max type changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        input data dtype = 'float32'
        clip_min = None
        clip_max = 1.0
    Changes:
        clip_max: float32 -> '1'
    Expected Results:
        The output of ClipGradByValue implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_min = None
    clip_max = "1"
    np_data, paddle_data = generate_test_data(length, shape)

    with pytest.raises(TypeError):
        paddle_clip = paddle.nn.ClipGradByValue(clip_max, clip_min)
        paddle_clip(paddle_data)
