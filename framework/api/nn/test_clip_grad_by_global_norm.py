#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clip_grad_by_global_norm
"""

from apibase import APIBase
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


def generate_test_data(length, shape):
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


class TestClipGradByGlobalNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]


obj = TestClipGradByGlobalNorm(paddle.nn.ClipGradByGlobalNorm)


@pytest.mark.api_nn_ClipGradByGlobalNorm_vartype
def test_clip_grad_by_global_norm_base():
    """
    base
    """
    shape = [10, 10]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm1():
    """
    input shape
    """
    shape = [9, 13, 11]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm2():
    """
    clip_norm
    """
    shape = [10, 10]
    length = 4
    clip_norm = -1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm3():
    """
    group_name
    """
    shape = [10, 10]
    length = 4
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm, group_name="test_group")
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByGlobalNorm_parameters
def test_clip_grad_by_global_norm1():
    """
    value
    """
    shape = [10, 10]
    length = 15
    clip_norm = 1.0
    np_data, paddle_data = generate_test_data(length, shape)
    np_res = numpy_clip_grad_by_global_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])
