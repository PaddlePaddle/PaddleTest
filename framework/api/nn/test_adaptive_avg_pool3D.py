#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.AdaptiveAvgPool3D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAdaptiveAvgPool3D(APIBase):
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


obj = TestAdaptiveAvgPool3D(paddle.nn.AdaptiveAvgPool3D)


def adaptive_start_index(index, input_size, output_size):
    """adaptive_start_index
    """
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    """adaptive_end_index
    """
    return int(np.ceil((index + 1) * input_size / output_size))


def compute_adaptive_pool3D(x, output_size, adaptive=True, data_format="NCDHW", pool_type="avg"):
    """
    adaptive_pool3d
    """
    N = x.shape[0]
    C, D, H, W = (
        [x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        if data_format == "NCDHW"
        else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]
    )

    if isinstance(output_size, int) or output_size is None:
        H_out = output_size
        W_out = output_size
        D_out = output_size
        output_size = [D_out, H_out, W_out]
    else:
        D_out, H_out, W_out = output_size

    if output_size[0] is None:
        output_size[0] = D
        D_out = D
    if output_size[1] is None:
        output_size[1] = H
        H_out = H
    if output_size[2] is None:
        output_size[2] = W
        W_out = W

    out = np.zeros((N, C, D_out, H_out, W_out)) if data_format == "NCDHW" else np.zeros((N, D_out, H_out, W_out, C))
    for k in range(D_out):
        d_start = adaptive_start_index(k, D, output_size[0])
        d_end = adaptive_end_index(k, D, output_size[0])

        for i in range(H_out):
            h_start = adaptive_start_index(i, H, output_size[1])
            h_end = adaptive_end_index(i, H, output_size[1])

            for j in range(W_out):
                w_start = adaptive_start_index(j, W, output_size[2])
                w_end = adaptive_end_index(j, W, output_size[2])

                if data_format == "NCDHW":
                    x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    if pool_type == "avg":
                        field_size = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
                        out[:, :, k, i, j] = np.sum(x_masked, axis=(2, 3, 4)) / field_size
                    elif pool_type == "max":
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == "NDHWC":
                    x_masked = x[:, d_start:d_end, h_start:h_end, w_start:w_end, :]
                    if pool_type == "avg":
                        field_size = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
                        out[:, k, i, j, :] = np.sum(x_masked, axis=(1, 2, 3)) / field_size
                    elif pool_type == "max":
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))
    return out


@pytest.mark.api_nn_AdaptiveAvgPool3D_vartype
def test_adaptive_avg_pool3D_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = [3, 3, 3]
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.base(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = [1, 1, 1]
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D1():
    """
    output_size = [2, 3, 3]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = [2, 3, 3]
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D2():
    """
    output_size = [2, 2, 2]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = [2, 2, 2]
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D3():
    """
    output_size = [1, 3, 2]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = [1, 3, 2]
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D4():
    """
    output_size = int(3)
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = 3
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D5():
    """
    output_size = tuple
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = (3, 3, 3)
    res = compute_adaptive_pool3D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D6():
    """
    output_size = tuple data_format="NCDHW"
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = (3, 3, 3)
    data_format = "NCDHW"
    res = compute_adaptive_pool3D(x=x, output_size=output_size, data_format=data_format)
    obj.run(res=res, data=x, output_size=output_size, data_format=data_format)


@pytest.mark.api_nn_AdaptiveAvgPool3D_parameters
def test_adaptive_avg_pool3D7():
    """
    output_size = tuple data_format="NDHWC"
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = (3, 3, 3)
    data_format = "NDHWC"
    res = compute_adaptive_pool3D(x=x, output_size=output_size, data_format=data_format)
    obj.run(res=res, data=x, output_size=output_size, data_format=data_format)


@pytest.mark.api_nn_AdaptiveAvgPool3D_exception
def test_adaptive_avg_pool3D8():
    """
    exception data_format="wrong"
    """
    x = randtool("float", -10, 10, [2, 3, 8, 32, 32])
    output_size = (3, 3, 3)
    data_format = "wrong"
    compute_adaptive_pool3D(x=x, output_size=output_size, data_format=data_format)
    obj.exception(etype=ValueError, mode="python", data=x, output_size=output_size, data_format=data_format)
