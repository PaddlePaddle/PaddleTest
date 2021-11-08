#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_adaptive_max_pool1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalAdaptiveMaxPool1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalAdaptiveMaxPool1d(paddle.nn.functional.adaptive_max_pool1d)


def adaptive_start_index(index, input_size, output_size):
    """
    adaptive_start_index
    """
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    """
    adaptive_start_index
    """
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool1D_forward_naive(
    x, ksize, strides, paddings, global_pool=0, ceil_mode=False, adaptive=False, data_type=np.float64
):
    """
    max_pool1D_forward_naive
    """
    N, C, L = x.shape
    if isinstance(ksize, int):
        ksize = [ksize]
    if isinstance(strides, int):
        strides = [strides]
    if isinstance(paddings, int):
        paddings = [paddings]
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (
            (L - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (L - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        out[:, :, i] = np.max(x_masked, axis=(2))
    return out


@pytest.mark.api_nn_adaptive_max_pool1d_vartype
def test_adaptive_max_pool1d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    output_size = 4
    res = max_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.base(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_max_pool1d_parameters
def test_adaptive_max_pool1d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 32])
    output_size = 8
    res = max_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.run(res=res, x=x, output_size=output_size)


# def test_adaptive_max_pool1d1():
#     """
#     output_size is list
#     """
#     x = randtool("float", -10, 10, [2, 3, 32])
#     output_size = [8]
#     res = max_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
#     obj.run(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_max_pool1d_parameters
def test_adaptive_max_pool1d2():
    """
    output_size is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 32])
    output_size = 8
    res = max_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.run(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_max_pool1d_parameters
def test_adaptive_max_pool1d3():
    """
    exception return_mask=True
    """
    obj2 = TestFunctionalAdaptiveMaxPool1d(paddle.nn.functional.adaptive_max_pool1d)
    obj2.static = False
    np.random.seed(22)
    x = np.random.rand(1, 1, 4)
    res = np.array([[[[0.20846054, 0.48168106, 0.42053804, 0.85918200]]], [[[0, 1, 2, 3]]]])
    obj2.run(res=res, x=x, output_size=4, return_mask=True)
