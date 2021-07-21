#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test topk
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np
import numpy.testing as npt


if paddle.device.is_compiled_with_cuda():
    devices = ['gpu', 'cpu']
else:
    devices = ['cpu']

types = [np.float32, np.float64, np.int32, np.int64]


@pytest.mark.api_base_topk_vartype
def test_topk_base():
    """
    x.shape=(2, 4)
    k = 1
    axis = -1
    """
    for t in types:
        x = np.array([[5.2, 7.1, 4.4, 1.],
                      [2.1, 2.5, 6.3, 5.4]]).astype(t)
        for d in devices:
            paddle.set_device(d)
            k = 1
            axis = -1
            value_temp, indices_temp = paddle.topk(paddle.to_tensor(x), k=k, axis=axis)
            value_x = value_temp.numpy()
            indices_x = indices_temp.numpy()
            value = np.array([[x[0, 1]], [x[1, 2]]]).astype(np.float64)
            indices = np.array([[1], [2]])
            npt.assert_allclose(value_x, value)
            npt.assert_equal(indices_x, indices)


@pytest.mark.api_base_topk_parameters
def test_topk1():
    """
    x.shape=(3, 4)
    k = 2
    axis = 0
    """
    x = np.array([[5.2, 7.1, 4.4, 1.],
                  [2.1, 2.5, 6.3, 5.4],
                  [3.5, 1.1, 8.3, 2.2]]).astype(np.float32)
    k = 2
    axis = 0
    value_temp, indices_temp = paddle.topk(paddle.to_tensor(x), k=k, axis=axis)
    value_x = value_temp.numpy()
    indices_x = indices_temp.numpy()
    value = np.array([[5.2, 7.1, 8.3, 5.4],
                      [3.5, 2.5, 6.3, 2.2]]).astype(np.float64)
    indices = np.array([[0, 0, 2, 1],
                        [2, 1, 1, 2]])
    npt.assert_allclose(value_x, value)
    npt.assert_equal(indices_x, indices)


@pytest.mark.api_base_topk_parameters
def test_topk2():
    """
    x.shape=(2, 2, 1)
    k = 1
    axis = -1
    """
    x = np.array([[[5, 7, 4, 1],
                   [2, 2, 6, 5]],
                  [[1, 5, 3, 2],
                   [3, 9, 8, 7]]]).astype(np.int32)
    k = 1
    axis = -1
    value_temp, indices_temp = paddle.topk(paddle.to_tensor(x), k=k, axis=axis)
    value_x = value_temp.numpy()
    indices_x = indices_temp.numpy()
    value = np.array([[[7], [6]],
                      [[5], [9]]]).astype(np.int32)
    indices = np.array([[[1], [2]],
                        [[1], [1]]])
    npt.assert_allclose(value_x, value)
    npt.assert_equal(indices_x, indices)


@pytest.mark.api_base_topk_parameters
def test_topk3():
    """
    x.shape=(2, 2, 1)
    k = 1
    axis = 2
    """
    x = np.array([[[5, 7, 4, 1],
                   [2, 2, 6, 5]],
                  [[1, 5, 3, 2],
                   [3, 9, 8, 7]]]).astype(np.int64)
    k = 1
    axis = 2
    value_temp, indices_temp = paddle.topk(paddle.to_tensor(x), k=k, axis=axis)
    value_x = value_temp.numpy()
    indices_x = indices_temp.numpy()
    value = np.array([[[7], [6]],
                      [[5], [9]]]).astype(np.int64)
    indices = np.array([[[1], [2]],
                        [[1], [1]]])
    npt.assert_allclose(value_x, value)
    npt.assert_equal(indices_x, indices)
