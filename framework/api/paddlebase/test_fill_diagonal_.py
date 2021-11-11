#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fill_diagonal_
"""

import pytest
import numpy as np
import paddle
import paddle.device as device

# global params
types = [np.float32, np.float64, np.int32, np.int64]
if device.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


def fill_diagonal_base(x, value, offset=0, warp=False):
    """
    api calculate
    """
    outputs, gradients = [], []
    for place in places:
        for t in types:
            paddle.disable_static(place)
            y = x.astype(t)

            y = paddle.to_tensor(y)
            y.stop_gradient = False
            y = y * 2
            out = paddle.Tensor.fill_diagonal_(y, value, offset, warp)
            outputs.append(out.numpy())
            loss = paddle.sum(out)
            loss.backward()
            gradients.append(y.grad.numpy())
    return outputs, gradients


@pytest.mark.api_base_fill_diagonal_vartype
def test_fill_diagonal_base():
    """
    base
    """
    x = np.zeros((3, 3))
    out, grad = fill_diagonal_base(x, 1)
    res_out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    res_grad = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    length = len(out)
    for i in range(length):
        assert np.allclose(out[i], res_out)
        assert np.allclose(grad[i], res_grad)


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_0():
    """
    default: wrap = False
    """
    x = np.zeros((5, 3))
    out, grad = fill_diagonal_base(x, 1)
    res_out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    res_grad = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(grad[0], res_grad)


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_1():
    """
    offset = 1
    value = 4
    """
    x = np.zeros((3, 3))
    out, grad = fill_diagonal_base(x, 4, offset=1)
    res_out = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    res_grad = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(grad[0], res_grad)


# @pytest.mark.api_base_fill_diagonal_parameters
# def test_fill_diagonal_2():
#     """
#     offset = -1
#     value = -4
#     """
#     x = np.zeros((3, 3))
#     out, grad = fill_diagonal_base(x, -4, offset=-1)
#     res_out = np.array([[0., 0., 0.],
#                         [-4., 0., 0.],
#                         [0., -4., 0.]])
#     res_grad = np.array([[1., 1., 1.],
#                          [0., 1., 1.],
#                          [1., 0., 1.]])
#
#     assert np.allclose(out[0], res_out)
#     assert np.allclose(grad[0], res_grad)


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_3():
    """
    wrap = True
    """
    x = np.zeros((7, 3))
    out, grad = fill_diagonal_base(x, 4, warp=True)
    res_out = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    res_grad = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    assert np.allclose(out[0], res_out)
    assert np.allclose(grad[0], res_grad)


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_4():
    """
    default: Multidimensional
    all dimensions of input must be of equal length
    """
    x = np.zeros((2, 2, 2))
    out, grad = fill_diagonal_base(x, 1)
    res_out = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    res_grad = np.array([[[0.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(grad[0], res_grad)
