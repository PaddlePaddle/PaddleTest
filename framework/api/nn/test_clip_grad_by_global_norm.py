#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clip_grad_by_global_norm
"""

from apibase import randtool, compare
import paddle
import pytest
import numpy as np


def numpy_clip_grad_by_global_norm(test_data, clip_norm):
    """
    ClipGradByGlobalNorm implemented by numpy.
    """
    cliped_data = []
    grad_data = []
    for data, grad in test_data:
        grad_data.append(grad)
    global_norm = np.sqrt(np.sum(np.square(np.array(grad_data))))
    if global_norm > clip_norm:
        for data, grad in test_data:
            grad = grad * clip_norm / global_norm
            cliped_data.append((data, grad))
    else:
        cliped_data = test_data
    return cliped_data


def generate_test_data(length, shape, value=10):
    """
    generate test data
    """
    tensor_data = []
    numpy_data = []
    np.random.seed(100)
    for i in range(length):
        np_weight = randtool("float", -value, value, shape)
        np_weight_grad = randtool("float", -value, value, shape)
        numpy_data.append((np_weight, np_weight_grad))

        tensor_weight = paddle.to_tensor(np_weight)
        tensor_weight_grad = paddle.to_tensor(np_weight_grad)
        tensor_data.append((tensor_weight, tensor_weight_grad))
    return numpy_data, tensor_data


@pytest.mark.api_nn_ClipGradByGlobalNorm_vartype
def test_clip_grad_by_global_norm_base():
    """
    Test base.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        clip_norm = 1.0
        value range: [-10, 10]

    Expected Results:
        The output of ClipGradByGlobalNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape, value=10)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm1():
    """
    Test ClipGradByGlobalNorm when input shape changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        input grad shape: [10, 10] -> [9, 13, 11]

    Expected Results:
        The output of ClipGradByGlobalNorm implemented by numpy and paddle should be equal.
    """
    shape = [9, 13, 11]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape, value=10)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm2():
    """
    Test ClipGradByGlobalNorm when clip_norm changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        clip_norm: 1.0 -> -1.0

    Expected Results:
        The output of ClipGradByGlobalNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_norm = -1.0
    np_data, paddle_data = generate_test_data(length, shape, value=10)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm3():
    """
    Test ClipGradByGlobalNorm when set group_name.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        group_name: 'test_group'

    Expected Results:
        The output of ClipGradByGlobalNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape, value=10)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm, group_name="test_group")
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm4():
    """
    Test ClipGradByGlobalNorm when value range changes.

    Test base config:
        input grad shape = [10, 10]
        input grad number = 4
        clip_norm = 1.0
        value range: [-10, 10]

    Changes:
        value range: [-10, 10] -> [-255555, 255555]

    Expected Results:
        The output of ClipGradByGlobalNorm implemented by numpy and paddle should be equal.
    """
    shape = [10, 10]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape, value=255555)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    # compare grad value computed by numpy and paddle
    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])
