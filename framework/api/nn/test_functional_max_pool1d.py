#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_max_pool1d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalMaxpool1d(APIBase):
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


obj = TestFunctionalMaxpool1d(paddle.nn.functional.max_pool1d)


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


@pytest.mark.api_nn_max_pool1d_vartype
def test_max_pool1d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 2
    padding = 0
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.base(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 3
    stride = 4
    padding = 0
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d1():
    """
    kernel_size = 1 stride = 1
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 1
    stride = 1
    padding = 0
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d2():
    """
    padding = 1
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 1
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d3():
    """
    ceil_mode = True
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 0
    ceil_mode = True
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding, ceil_mode=ceil_mode)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d4():
    """
    exception return_mask=True
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 1
    # res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.exception(
        etype=AttributeError,
        mode="python",
        x=x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        return_mask=True,
    )


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d5():
    """
    padding is list
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = [1]
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d6():
    """
    padding is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = 1
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d7():
    """
    stride is list
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = [1]
    padding = 1
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d8():
    """
    stride is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 2
    padding = 1
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d9():
    """
    kernel_size is list
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = [3]
    stride = 1
    padding = 1
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)


@pytest.mark.api_nn_max_pool1d_parameters
def test_max_pool1d10():
    """
    kernel_size is tuple, padding list BUG!!!
    """
    x = randtool("float", -10, 10, [2, 3, 8])
    kernel_size = 2
    stride = 1
    padding = [1, 1]
    res = max_pool1D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=stride, padding=padding)
