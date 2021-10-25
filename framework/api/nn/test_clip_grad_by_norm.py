#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clip_grad_by_norm
"""

from apibase import randtool, compare, APIBase
import paddle
import pytest
import numpy as np


def numpy_clip_grad_by_norm(test_data, clip_norm):
    """
    ClipGradByNorm implemented by numpy.
    """
    cliped_data = []
    for data, grad in test_data:
        norm = np.sqrt(np.sum(np.square(np.array(grad))))
        if norm > clip_norm:
            grad = grad * clip_norm / norm
        cliped_data.append((data, grad))
    return cliped_data


def generate_test_data(length, shape, dtype, value=10):
    """
    generate test data
    """
    tensor_data = []
    numpy_data = []
    np.random.seed(100)
    for i in range(length):
        np_weight = randtool("float", -value, value, shape).astype(dtype)
        np_weight_grad = randtool("float", -value, value, shape).astype(dtype)
        numpy_data.append((np_weight, np_weight_grad))

        tensor_weight = paddle.to_tensor(np_weight)
        tensor_weight_grad = paddle.to_tensor(np_weight_grad)
        tensor_data.append((tensor_weight, tensor_weight_grad))
    return numpy_data, tensor_data


@pytest.mark.api_nn_ClipGradByNorm_vartype
def test_clip_grad_by_norm_base():
    """
    Test base.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Expected Results:
        The output of ClipGradByNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm1():
    """
    Test ClipGradByNorm when input shape changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        input grad shape: [10, 10] -> [7, 13, 10]

    Expected Results:
        The output of ClipGradByNorm implemented by numpy and paddle should be equal.
    """
    shape = [7, 13, 10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm2():
    """
    Test ClipGradByNorm when input shape changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        input grad shape: [10, 10] -> [10]

    Expected Results:
        The output of ClipGradByNorm implemented by numpy and paddle should be equal.
    """
    shape = [10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm3():
    """
    Test ClipGradByNorm when clip_norm changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        clip_norm: -1.0

    Expected Results:
        clip_norm cann't less equal than 0.0, raise ValueError.
    """
    shape = [10, 10]
    length = 5
    clip_norm = -1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)

    with pytest.raises(ValueError):
        paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        paddle_clip(paddle_data)


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm4():
    """
    Test ClipGradByNorm when value range changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        value range: [-10, 10] -> [-25555, 25555]

    Expected Results:
        The output of ClipGradByNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=25555)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm5():
    """
    Test whether clip_norm is constrained.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        clip_norm: 1.0 -> 0.01

    Expected Results:
        'clip_norm' should be constrained when l2 norm is bigger than 'clip_norm'.
    """
    shape = [10, 10]
    length = 5
    clip_norm = 0.01
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_clip(paddle_data)
    compare(clip_norm, paddle_clip.clip_norm)  # clip_norm is constrained


@pytest.mark.api_nn_ClipGradByNorm_vartype
def test_clip_grad_by_norm6():
    """
    Test unsupport input grad dtype.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 5
        input data type = 'float32'
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        input grad type: 'float32' -> ['int8', 'int16', 'int32', 'float16', 'float64']

    Expected Results:
        Raise RuntimeError when input grad type is not supported.
    """
    shape = [10, 10]
    length = 5
    clip_norm = 1.0
    unsupport_dtype = ["int8", "int16", "int32", "float16", "float64"]

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    for dtype in unsupport_dtype:
        np_data, paddle_data = generate_test_data(length, shape, dtype, value=10)
        try:
            paddle_clip(paddle_data)
        except RuntimeError:
            pass
