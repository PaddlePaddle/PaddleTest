#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_unfold
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestUnfold(APIBase):
    """
    test unfold
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestUnfold(paddle.nn.functional.unfold)


def im2col(input_data, kh, kw, stride=1, pad=0, dilation=1):
    """
    calculate im2col
    """
    N, C, H, W = input_data.shape
    dh, dw = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1
    h_out = (H + 2 * pad - dh) // stride + 1
    w_out = (W + 2 * pad - dw) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant", constant_values=0)
    col = np.zeros((N, C, dh, dw, h_out, w_out))

    for y in range(dh):
        y_max = y + stride * h_out
        for x in range(dw):
            x_max = x + stride * w_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    res = col.reshape((N, C * dh * dw, h_out * w_out))
    return res


@pytest.mark.api_nn_unfold_vartype
def test_flatten_base():
    """
    base
    """
    x = np.random.rand(2, 3, 10, 10)
    res = im2col(x, 3, 3)
    obj.base(res=res, x=x, kernel_sizes=3)


@pytest.mark.api_nn_unfold_parameters
def test_flatten0():
    """
    default
    """
    x = np.random.rand(2, 3, 10, 10)
    res = im2col(x, 3, 3)
    obj.run(res=res, x=x, kernel_sizes=[3, 3])


@pytest.mark.api_nn_unfold_parameters
def test_flatten1():
    """
    paddings=1
    """
    x = np.random.rand(2, 3, 10, 10)
    res = im2col(x, 3, 3, pad=1)
    obj.run(res=res, x=x, kernel_sizes=[3, 3], paddings=1)


@pytest.mark.api_nn_unfold_parameters
def test_flatten2():
    """
    kh != kw
    paddings=1
    """
    x = np.random.rand(2, 3, 10, 10)
    res = im2col(x, 2, 4, pad=1)
    obj.run(res=res, x=x, kernel_sizes=[2, 4], paddings=1)


@pytest.mark.api_nn_unfold_parameters
def test_flatten3():
    """
    kh != kw
    paddings=1
    stride=2
    """
    x = np.random.rand(2, 3, 10, 10)
    res = im2col(x, 2, 4, pad=1, stride=2)
    obj.run(res=res, x=x, kernel_sizes=[2, 4], paddings=1, strides=2)
