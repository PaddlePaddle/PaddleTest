#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test dygraph setitem
"""
import paddle
import numpy as np
from apibase import compare


def test_ellipsis_param():
    """
    x.shape=[2, 6]
    x[...] = 42
    dtype="float64"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.arange(0, 12).reshape(2, 6).astype(np.float32))
    tensor_x[...] = 42
    res = tensor_x.numpy()
    exp = np.array([[42.0, 42.0, 42.0, 42.0, 42.0, 42.0], [42.0, 42.0, 42.0, 42.0, 42.0, 42.0]])
    compare(res, exp)


def test_ellipsis_array():
    """
    x.shape=[2, 6]
    x[...] = np.array
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.arange(0, 12).reshape(2, 6).astype(np.float32))
    tensor_x[...] = np.arange(21, 27).astype(np.float32)
    res = tensor_x.numpy()
    exp = np.array([[21.0, 22.0, 23.0, 24.0, 25.0, 26.0], [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]])
    compare(res, exp)


def test_ellipsis_tensor():
    """
    x.shape=[2, 6]
    x[...] = tensor
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.arange(0, 12).reshape(2, 6).astype(np.float32))
    tensor_y = paddle.to_tensor(np.arange(21, 27).astype(np.float32))
    tensor_x[...] = tensor_y
    res = tensor_x.numpy()
    exp = np.array([[21.0, 22.0, 23.0, 24.0, 25.0, 26.0], [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]])
    compare(res, exp)


def test_minus_step_param():
    """
    x.shape=[2, 6]
    x[1, 4::-1] = 42
    dtype="float64"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float64))
    tensor_x[1, 4::-1] = 42
    res = tensor_x.numpy()
    exp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [42.0, 42.0, 42.0, 42.0, 42.0, 0.0]])
    compare(res, exp)


def test_minus_step_array():
    """
    x.shape=[2, 6]
    x[:, 4:1:-1] = np.array
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
    tensor_x[:, 4:1:-1] = np.array([1, 3, 5]).astype(np.float32)
    res = tensor_x.numpy()
    exp = np.array([[0.0, 0.0, 5.0, 3.0, 1.0, 0.0], [0.0, 0.0, 5.0, 3.0, 1.0, 0.0]])
    compare(res, exp)


def test_minus_step_tensor():
    """
    x.shape=[2, 6]
    x[:, 3:0:-1] = tensor
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
    tensor_y = paddle.to_tensor(np.arange(21, 24).astype(np.float32))
    tensor_x[:, 3:0:-1] = tensor_y
    res = tensor_x.numpy()
    exp = np.array([[0.0, 23.0, 22.0, 21.0, 0.0, 0.0], [0.0, 23.0, 22.0, 21.0, 0.0, 0.0]])
    compare(res, exp)


def test_inter_step_param():
    """
    x.shape=[2, 6]
    x[1, 4::-2] = 42
    dtype="float64"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float64))
    tensor_x[1, 4::-2] = 42
    res = tensor_x.numpy()
    exp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [42.0, 0.0, 42.0, 0.0, 42.0, 0.0]])
    compare(res, exp)


def test_inter_step_array():
    """
    x.shape=[2, 6]
    x[:, 5:0:-2] = np.array
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
    tensor_x[:, 5:0:-2] = np.array([1, 3, 5]).astype(np.float32)
    res = tensor_x.numpy()
    exp = np.array([[0.0, 5.0, 0.0, 3.0, 0.0, 1.0], [0.0, 5.0, 0.0, 3.0, 0.0, 1.0]])
    compare(res, exp)


def test_inter_step_tensor():
    """
    x.shape=[2, 6]
    x[:, 5:0:-2] = tensor
    dtype="float32"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
    tensor_y = paddle.to_tensor(np.arange(21, 24).astype(np.float32))
    tensor_x[:, 5:0:-2] = tensor_y
    res = tensor_x.numpy()
    exp = np.array([[0.0, 23.0, 0.0, 22.0, 0.0, 21.0], [0.0, 23.0, 0.0, 22.0, 0.0, 21.0]])
    compare(res, exp)


# def test_tensor_param():
#     """
#     x.shape=[2, 6]
#     x[:, tensor:tensor] = 42
#     dtype="float32"
#     """
#     paddle.disable_static()
#     tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
#     tensor_y1 = paddle.zeros([1]) + 2
#     tensor_y2 = paddle.zeros([1]) + 5
#     tensor_x[:, tensor_y1:tensor_y2] = 42
#     res = tensor_x.numpy()
#     exp = np.array([[0., 0., 42., 42., 42., 0.],
#                     [0., 0., 42., 42., 42., 0.]])
#     compare(res, exp)


def test_tensor_array():
    """
    x.shape=[2, 6]
    x[tensor, 1:tensor] = np.array
    dtype="float64"
    """
    paddle.disable_static()
    tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float64))
    tensor_y1 = paddle.zeros([1]) + 1
    tensor_y2 = paddle.zeros([1]) + 3
    tensor_x[tensor_y1, 1:tensor_y2] = np.array([1, 3]).astype(np.float64)
    res = tensor_x.numpy()
    exp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0, 0.0, 0.0]])
    compare(res, exp)


# def test_tensor_tensor():
#     """
#
#     :return:
#     """
#     paddle.disable_static()
#     tensor_x = paddle.to_tensor(np.zeros(12).reshape(2, 6).astype(np.float32))
#     tensor_y1 = paddle.zeros([1]) + 5
#     tensor_y2 = paddle.zeros([1]) - 2
#     tensor_z = paddle.to_tensor(np.arange(21, 24).astype(np.float32))
#     tensor_x[:, tensor_y1:0:tensor_y2] = tensor_z
#     res = tensor_x.numpy()
#     exp = np.array([[0., 23., 0., 22., 0., 21.],
#                     [0., 23., 0., 22., 0., 21.]])
#     compare(res, exp)
