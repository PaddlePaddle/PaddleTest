#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_clip_grad_by_norm
"""

from apibase import APIBase
from apibase import randtool, compare
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


class TestClipGradByNorm(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]


obj = TestClipGradByNorm(paddle.nn.ClipGradByNorm)


@pytest.mark.api_nn_ClipGradByNorm_vartype
def test_clip_grad_by_norm_base():
    """
    base
    """
    shape = [10, 10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm1():
    """
    input shape
    """
    shape = [7, 13, 10]
    length = 5
    clip_norm = 1.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm2():
    """
    clip_norm
    """
    shape = [10, 10]
    length = 5
    clip_norm = 10.0
    dtype = "float32"
    np_data, paddle_data = generate_test_data(length, shape, dtype)
    np_res = numpy_clip_grad_by_norm(np_data, clip_norm=clip_norm)

    paddle_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    paddle_cliped_data = paddle_clip(paddle_data)
    paddle_res = []
    for w, g in paddle_cliped_data:
        paddle_res.append((w.numpy(), g.numpy()))

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])


@pytest.mark.api_nn_ClipGradByNorm_parameters
def test_clip_grad_by_norm3():
    """
    value
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

    for res, p_res in zip(np_res, paddle_res):
        compare(res[1], p_res[1])
