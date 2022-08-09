#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_adaptive_avg_pool1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalAdaptiveAvgPool1d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-4
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalAdaptiveAvgPool1d(paddle.nn.functional.adaptive_avg_pool1d)


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


def avg_pool1D_forward_naive(
    x, ksize, strides, paddings, global_pool=0, ceil_mode=False, exclusive=True, adaptive=False, data_type=np.float64
):
    """
    avg_pool1D_forward_naive
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

        field_size = (r_end - r_start) if (exclusive or adaptive) else (ksize[0])
        if data_type == np.int8 or data_type == np.uint8:
            out[:, :, i] = (np.rint(np.sum(x_masked, axis=(2, 3)) / field_size)).astype(data_type)
        else:
            out[:, :, i] = (np.sum(x_masked, axis=(2)) / field_size).astype(data_type)
    return out


@pytest.mark.api_nn_adaptive_avg_pool1d_vartype
def test_adaptive_avg_pool1d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    output_size = 2
    res = avg_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.base(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_avg_pool1d_parameters
def test_adaptive_avg_pool1d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    output_size = 2
    res = avg_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.run(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_avg_pool1d_parameters
def test_adaptive_avg_pool1d1():
    """
    output_size = 8
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    output_size = 8
    res = avg_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.run(res=res, x=x, output_size=output_size)


# def test_adaptive_avg_pool1d2():
#     """
#     output_size = [7]
#     """
#     x = randtool("float", -10, 10, [2, 3, 8])
#     output_size = [7,]
#     res = avg_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
#     obj.run(res=res, x=x, output_size=output_size)


@pytest.mark.api_nn_adaptive_avg_pool1d_parameters
def test_adaptive_avg_pool1d3():
    """
    output_size = (4)
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    output_size = 4
    res = avg_pool1D_forward_naive(x=x, ksize=output_size, strides=0, paddings=0, adaptive=True)
    obj.run(res=res, x=x, output_size=output_size)
