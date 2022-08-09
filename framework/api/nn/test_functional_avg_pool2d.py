#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_avg_pool2d
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalAvgPool2d(APIBase):
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


obj = TestFunctionalAvgPool2d(paddle.nn.functional.avg_pool2d)


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


def avg_pool2D_forward_naive(
    x, ksize, strides, paddings, global_pool=0, ceil_mode=False, exclusive=True, adaptive=False, data_type=np.float64
):
    """
    avg_pool2D_forward_naive
    """
    N, C, H, W = x.shape
    # a = strides[0] + 1 if ceil_mode else (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
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

            field_size = ((r_end - r_start) * (c_end - c_start)) if (exclusive or adaptive) else (ksize[0] * ksize[1])
            if data_type == np.int8 or data_type == np.uint8:
                out[:, :, i, j] = (np.rint(np.sum(x_masked, axis=(2, 3)) / field_size)).astype(data_type)
            else:
                out[:, :, i, j] = (np.sum(x_masked, axis=(2, 3)) / field_size).astype(data_type)
    return out


@pytest.mark.api_nn_avg_pool2d_vartype
def test_avg_pool2d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [2, 2]
    strides = kernel_size
    padding = [0, 0, 0, 0]
    res = avg_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.base(res=res, x=x, kernel_size=kernel_size)


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 32, 32])
    kernel_size = [3, 3]
    strides = kernel_size
    padding = [0, 0, 0, 0]
    res = avg_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size)


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d1():
    """
    ceil_mode=True
    """
    np.random.seed(33)
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = kernel_size
    padding = [0, 0]
    ceil_mode = True
    res = np.array(
        [
            [
                [[-0.4076, 0.7130], [-6.0360, 9.6633]],
                [[0.2288, -5.6701], [5.2855, 6.4314]],
                [[-1.9028, -3.6577], [3.1812, -5.1751]],
            ],
            [
                [[-1.9191, -5.7106], [-0.1996, -2.0623]],
                [[2.3765, 1.2110], [0.3886, 0.2039]],
                [[1.7489, 0.9140], [-0.2966, 9.2774]],
            ],
        ]
    )
    obj.run(
        res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=False
    )


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d2():
    """
    exclusive = False
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = kernel_size
    padding = [0, 0, 0, 0]
    ceil_mode = False
    exclusive = False
    res = avg_pool2D_forward_naive(
        x=x, ksize=kernel_size, strides=strides, paddings=padding, ceil_mode=ceil_mode, exclusive=exclusive
    )
    obj.run(
        res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding, ceil_mode=ceil_mode, exclusive=exclusive
    )


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d3():
    """
    strides = [1, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = [1, 1]
    padding = [0, 0]
    res = avg_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding)


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d4():
    """
    padding = [1, 1] strides=[1, 1] bug!!!!!!!!
    """
    np.random.seed(33)
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = [1, 1]
    padding = [1, 1]
    exclusive = False
    res = np.array(
        [
            [
                [
                    [-5.4684e-01, -1.8122e00, -1.6018e00, -7.9077e-01],
                    [-1.7564e-01, -4.0758e-01, -8.3515e-01, 5.7322e-03],
                    [-7.7013e-01, -1.5516e00, 2.6144e-03, 1.0625e00],
                    [-8.9332e-01, -6.0741e-01, 7.6272e-01, 1.1227e00],
                ],
                [
                    [7.6297e-01, 8.1572e-01, -1.4238e00, -8.5005e-01],
                    [-8.3171e-01, 2.2876e-01, -2.3339e00, -8.2958e-01],
                    [-9.4761e-01, 1.3780e00, 2.6285e-01, 1.2584e00],
                    [-8.7794e-01, 1.1749e00, 1.3451e00, 1.7802e00],
                ],
                [
                    [-9.3170e-01, -1.7112e00, -2.0753e00, -9.5809e-01],
                    [-1.5909e00, -1.9028e00, -3.0021e00, -1.5312e00],
                    [-8.2433e-01, -2.0466e-01, -8.2089e-01, -2.3907e-01],
                    [-2.0591e-01, 8.6880e-01, -3.8009e-01, -5.4096e-01],
                ],
            ],
            [
                [
                    [-5.6600e-01, -1.3322e00, -2.5635e00, -1.7563e00],
                    [-9.9274e-01, -1.9191e00, -4.3300e00, -2.8299e00],
                    [5.2522e-01, -9.6029e-01, -3.7981e00, -2.8683e00],
                    [-1.4690e-01, -6.5351e-01, -2.5792e00, -1.6491e00],
                ],
                [
                    [-8.7719e-01, 5.9388e-01, 8.9107e-01, 1.4612e00],
                    [1.0631e00, 2.3765e00, 2.1954e00, 1.7170e00],
                    [2.4809e00, 2.7749e00, 2.6821e00, 1.4724e00],
                    [2.2153e00, 1.9121e00, 1.8426e00, 1.3298e-01],
                ],
                [
                    [-1.9444e-01, -4.7598e-01, -4.4598e-01, 7.5427e-01],
                    [9.7965e-01, 1.7489e00, 8.6127e-03, 1.0739e00],
                    [6.7954e-01, 1.8952e00, 1.4729e00, 1.6663e00],
                    [7.3454e-01, 2.1260e00, 1.9734e00, 1.6912e00],
                ],
            ],
        ]
    )
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding, exclusive=exclusive)


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d5():
    """
    strides = tuple
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = (1, 1)
    padding = [0, 0]
    res = avg_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding)


@pytest.mark.api_nn_avg_pool2d_parameters
def test_avg_pool2d6():
    """
    padding = tuple
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    kernel_size = [3, 3]
    strides = (1, 1)
    padding = (0, 0)
    res = avg_pool2D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=padding)
    obj.run(res=res, x=x, kernel_size=kernel_size, stride=strides, padding=padding)
