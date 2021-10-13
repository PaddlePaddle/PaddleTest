#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fill_diagonal_tensor
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


def fill_diagonal_tensor_base(x, y, offset=0, dim1=0, dim2=1):
    """
    api calculate
    """
    outputs, x_gradients, y_gradients = [], [], []
    for place in places:
        for t in types:
            paddle.disable_static(place)
            x = x.astype(t)
            y = y.astype(t)
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.Tensor.fill_diagonal_tensor(x, y, offset, dim1, dim2)
            outputs.append(out.numpy())
            loss = paddle.sum(out)
            loss.backward()
            x_gradients.append(x.grad.numpy())
            y_gradients.append(y.grad)
    return outputs, x_gradients, y_gradients


@pytest.mark.api_base_fill_diagonal_vartype
def test_fill_diagonal_base():
    """
    base
    """
    x = np.zeros((3, 3))
    y = np.ones((3,))
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y)
    res_out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    res_grad = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    length = len(out)
    for i in range(length):
        assert np.allclose(out[i], res_out)
        assert np.allclose(x_grad[i], res_grad)
        assert y_grad[i] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_0():
    """
    default
    """
    x = np.zeros((5, 3))
    y = np.ones((3,))
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y)
    res_out = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    res_grad = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_1():
    """
    offset = 1
    """
    x = np.zeros((3, 3))
    y = np.ones((2,)) * 4
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y, offset=1)
    res_out = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    res_grad = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_2():
    """
    offset = -1
    value = -4
    """
    x = np.zeros((3, 3))
    y = np.ones((2,)) * -4
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y, offset=-1)
    res_out = np.array([[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, -4.0, 0.0]])
    res_grad = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_3():
    """
    offset = -1
    value = -4
    """
    x = np.zeros((3, 3))
    y = np.ones((1,)) * -4
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y, offset=-2)
    res_out = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    res_grad = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_4():
    """
    default: Multidimensional
    """
    x = np.zeros((2, 4, 4))
    y = np.ones((4, 2))
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y)
    res_out = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    res_grad = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_5():
    """
    Multidimensional
    dim1 = 1, dim2 = 2
    """
    x = np.zeros((2, 4, 7))
    y = np.ones((2, 4))
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y, dim1=1, dim2=2)
    res_out = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    res_grad = np.array(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None


@pytest.mark.api_base_fill_diagonal_parameters
def test_fill_diagonal_6():
    """
    Multidimensional
    dim1 = 1, dim2 = -1
    """
    x = np.zeros((2, 4, 7))
    y = np.ones((2, 4))
    out, x_grad, y_grad = fill_diagonal_tensor_base(x, y, dim1=1, dim2=2)
    res_out = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    res_grad = np.array(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    assert np.allclose(out[0], res_out)
    assert np.allclose(x_grad[0], res_grad)
    assert y_grad[0] is None
