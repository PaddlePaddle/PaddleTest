#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_max_pool2d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalMaxPool2d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalMaxPool2d(paddle.nn.functional.max_pool2d)


def adaptive_start_index(index, input_size, output_size):
    """
    adaptive_start_index
    """
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    """
    adaptive_end_index
    """
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool2D_forward_naive(
    x, ksize, strides, paddings, global_pool=0, ceil_mode=False, exclusive=True, adaptive=False, data_type=np.float64
):
    """
    max_pool2D_forward_naive
    """
    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (
            (H - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )
        W_out = (
            (W - ksize[1] + 2 * paddings[1] + strides[1] - 1) // strides[1] + 1
            if ceil_mode
            else (W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        )
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            if adaptive:
                r_start = adaptive_start_index(i, H, ksize[0])
                r_end = adaptive_end_index(i, H, ksize[0])
                c_start = adaptive_start_index(j, W, ksize[1])
                c_end = adaptive_end_index(j, W, ksize[1])
            else:
                r_start = np.max((i * strides[0] - paddings[0], 0))
                r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
                c_start = np.max((j * strides[1] - paddings[1], 0))
                c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
    return out


@pytest.mark.api_nn_max_pool2d_vartype
def test_max_pool2d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [2, 2]
    strides = kernel_size
    padding = [0, 0, 0, 0]
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.base(res=res, x=x, kernel_size=kernel_size)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = kernel_size
    padding = [0, 0, 0, 0]
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d1():
    """
    ceil_mode=True
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = [1, 1]
    padding = [0, 0, 0, 0]
    ceil_mode = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding, ceil_mode=ceil_mode)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides, ceil_mode=ceil_mode)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d2():
    """
    strides = [1, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = [1, 1]
    padding = [0, 0, 0, 0]
    # return_indices = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d3():
    """
    strides = [1, 1], padding = [1, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = [1, 1]
    padding = [1, 1]
    # return_indices = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, padding=padding, stride=strides)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d4():
    """
    strides = [1, 2]
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = [1, 2]
    padding = [0, 0]
    # return_indices = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d5():
    """
    strides is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = (1, 2)
    padding = [0, 0]
    # return_indices = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides)


@pytest.mark.api_nn_max_pool2d_parameters
def test_max_pool2d6():
    """
    padding is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = (1, 2)
    padding = (0, 0)
    # return_indices = True
    res = max_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides)
