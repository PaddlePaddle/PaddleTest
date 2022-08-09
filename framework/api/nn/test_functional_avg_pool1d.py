#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_avg_pool1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalAvgPool1d(APIBase):
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


obj = TestFunctionalAvgPool1d(paddle.nn.functional.avg_pool1d)


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


@pytest.mark.api_nn_avg_pool1d_vartype
def test_avg_pool1d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 2
    padding = 0
    res = avg_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.base(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_avg_pool1d_parameters
def test_avg_pool1d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 3
    stride = 4
    padding = 0
    res = avg_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_avg_pool1d_parameters
def test_avg_pool1d1():
    """
    kernel_size = 1 stride = 1
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 1
    stride = 1
    padding = 0
    res = avg_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_avg_pool1d_parameters
def test_avg_pool1d2():
    """
    padding = 1
    """
    np.random.seed(33)
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 1
    exclusive = False
    res = np.array(
        [
            [
                [-2.5149, -3.0151, -1.3908, -3.2876, 1.3070, 0.5544, -7.9530, -0.2709, 4.5325],
                [1.8045, 1.6704, 4.5161, 3.5843, -5.2704, -5.6904, -4.8496, 1.4680, 4.8317],
                [3.8063, 3.7469, -1.0498, -1.4775, 1.7217, -0.3135, -1.2945, -2.3477, -3.5755],
            ],
            [
                [-2.9882, -7.1761, 0.3469, 0.0921, -3.4473, 3.2253, 6.9329, 7.9186, 3.2157],
                [0.2755, -1.4097, -3.1454, -5.6699, -3.6506, -2.7829, -5.3897, 1.3585, 3.4061],
                [-1.3744, -2.9662, 0.5122, -2.5789, -4.9590, 2.0396, 5.0479, 0.1446, -2.5876],
            ],
        ]
    )
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=exclusive)


@pytest.mark.api_nn_avg_pool1d_parameters
def test_avg_pool1d3():
    """
    ceil_mode = True
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 0
    ceil_mode = True
    res = avg_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding, ceil_mode=ceil_mode)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
