#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_unbind


"""
import paddle
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.api_base_unbind_parameters
def test_unbind():
    """
    x.shape = (3, 2, 2)
    axis = 0
    """
    x = np.arange(1, 13).reshape((3, 2, 2)).astype(np.int32)
    axis = 0
    x_tensor = paddle.to_tensor(x)
    out_list = paddle.unbind(x_tensor, axis=axis)
    length = len(out_list)
    for i in range(length):
        ept = x[i, :, :]
        npt.assert_allclose(out_list[i].numpy(), ept)


@pytest.mark.api_base_unbind_parameters
def test_unbind1():
    """
    x.shape = (4, 2, 3)
    axis = -1
    """
    x = np.arange(1, 25).reshape((4, 2, 3)).astype(np.float32)
    axis = -1
    x_tensor = paddle.to_tensor(x)
    out_list = paddle.unbind(x_tensor, axis=axis)
    length = len(out_list)
    for i in range(length):
        ept = x[:, :, i]
        npt.assert_allclose(out_list[i].numpy(), ept)


@pytest.mark.api_base_unbind_parameters
def test_unbind2():
    """
    x.shape = (4, 2, 3)
    axis = 2
    """
    x = np.arange(1, 25).reshape((4, 2, 3)).astype(np.float32)
    axis = 2
    x_tensor = paddle.to_tensor(x)
    out_list = paddle.unbind(x_tensor, axis=axis)
    length = len(out_list)
    for i in range(length):
        ept = x[:, :, i]
        npt.assert_allclose(out_list[i].numpy(), ept)


@pytest.mark.api_base_unbind_parameters
def test_unbind3():
    """
    x.shape = (4, 3, 2, 2)
    axis = 1
    """
    x = np.arange(1, 49).reshape((4, 3, 2, 2)).astype(np.float64)
    axis = 1
    x_tensor = paddle.to_tensor(x)
    out_list = paddle.unbind(x_tensor, axis=axis)
    length = len(out_list)
    for i in range(length):
        ept = x[:, i, :, :]
        npt.assert_allclose(out_list[i].numpy(), ept)
