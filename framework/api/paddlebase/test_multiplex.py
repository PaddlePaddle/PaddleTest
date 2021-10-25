#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test multiplex
"""
import paddle
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.api_base_multiplex_parameters
def test_multiplex():
    """
    2 input x tensor
    x.shape = (4, 4)
    index = [[1], [0]]
    """
    x1 = np.arange(1, 17).reshape(4, 4).astype(np.float32)
    x2 = np.arange(-17, -1).reshape(4, 4).astype(np.float32)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    index = np.array([[1], [0]]).astype(np.int32)
    index = paddle.to_tensor(index)
    out = paddle.fluid.layers.multiplex(inputs=[x1, x2], index=index)
    res = paddle.to_tensor(
        np.array(
            [[-17.0, -16.0, -15.0, -14.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]
        ).astype(np.float32)
    )
    npt.assert_array_equal(out.numpy(), res.numpy())


@pytest.mark.api_base_multiplex_parameters
def test_multiplex1():
    """
    2 input x tensor
    x.shape = (4, 4)
    index = [[1], [0], [1], [1]]
    """
    x1 = np.arange(1, 17).reshape(4, 4).astype(np.float32)
    x2 = np.arange(-17, -1).reshape(4, 4).astype(np.float32)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    index = np.array([[1], [0], [1], [1]]).astype(np.int32)
    index = paddle.to_tensor(index)
    out = paddle.fluid.layers.multiplex(inputs=[x1, x2], index=index)
    res = paddle.to_tensor(
        np.array(
            [[-17.0, -16.0, -15.0, -14.0], [5.0, 6.0, 7.0, 8.0], [-9.0, -8.0, -7.0, -6.0], [-5.0, -4.0, -3.0, -2.0]]
        ).astype(np.float32)
    )
    npt.assert_array_equal(out.numpy(), res.numpy())


@pytest.mark.api_base_multiplex_parameters
def test_multiplex2():
    """
    2 input x tensor
    x.shape = (4, 4)
    index = [[1], [0], [1], [1], [0], [0]]
    """
    x1 = np.arange(1, 17).reshape(4, 4).astype(np.float32)
    x2 = np.arange(-17, -1).reshape(4, 4).astype(np.float32)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    index = np.array([[1], [0], [1], [1], [0], [0]]).astype(np.int32)
    index = paddle.to_tensor(index)
    out = paddle.fluid.layers.multiplex(inputs=[x1, x2], index=index)
    res = paddle.to_tensor(
        np.array(
            [[-17.0, -16.0, -15.0, -14.0], [5.0, 6.0, 7.0, 8.0], [-9.0, -8.0, -7.0, -6.0], [-5.0, -4.0, -3.0, -2.0]]
        ).astype(np.float32)
    )
    npt.assert_array_equal(out.numpy(), res.numpy())


@pytest.mark.api_base_multiplex_parameters
def test_multiplex3():
    """
    3 input x tensor
    x.shape = (2, 3)
    index = [[1], [2], [0]]
    """
    x1 = np.arange(1, 7).reshape(2, 3).astype(np.float32)
    x2 = np.arange(-7, -1).reshape(2, 3).astype(np.float32)
    x3 = np.arange(11, 17).reshape(2, 3).astype(np.float32)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    x3 = paddle.to_tensor(x3)
    index = np.array([[1], [2], [0]]).astype(np.int32)
    index = paddle.to_tensor(index)
    out = paddle.fluid.layers.multiplex(inputs=[x1, x2, x3], index=index)
    res = paddle.to_tensor(np.array([[-7.0, -6.0, -5.0], [14.0, 15.0, 16.0]]).astype(np.float32))
    npt.assert_array_equal(out.numpy(), res.numpy())


@pytest.mark.api_base_multiplex_parameters
def test_multiplex4():
    """
    3 input tensor
    x.shape = (3, 3)
    index = [[1], [2], [0]]
    """
    x1 = np.arange(1, 10).reshape(3, 3).astype(np.float32)
    x2 = np.arange(-10, -1).reshape(3, 3).astype(np.float32)
    x3 = np.arange(11, 20).reshape(3, 3).astype(np.float32)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    x3 = paddle.to_tensor(x3)
    index = np.array([[1], [2], [0]]).astype(np.int32)
    index = paddle.to_tensor(index)
    out = paddle.fluid.layers.multiplex(inputs=[x1, x2, x3], index=index)
    res = paddle.to_tensor(np.array([[-10.0, -9.0, -8.0], [14.0, 15.0, 16.0], [7.0, 8.0, 9.0]]).astype(np.float32))
    npt.assert_array_equal(out.numpy(), res.numpy())
