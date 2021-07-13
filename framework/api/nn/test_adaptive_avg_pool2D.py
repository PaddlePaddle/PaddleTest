#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.AdaptiveAvgPool2D
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAdaptiveAvgPool2D(APIBase):
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


obj = TestAdaptiveAvgPool2D(paddle.nn.AdaptiveAvgPool2D)


def adaptive_start_index(index, input_size, output_size):
    """adaptive_start_index
    """
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    """
    adaptive_end_index
    """
    return int(np.ceil((index + 1) * input_size / output_size))


def compute_adaptive_pool2D(x, output_size, data_format="NCHW", pool_type="avg"):
    """
    compute
    """
    N = x.shape[0]
    C, H, W = [x.shape[1], x.shape[2], x.shape[3]] if data_format == "NCHW" else [x.shape[3], x.shape[1], x.shape[2]]

    if isinstance(output_size, int) or output_size is None:
        H_out = output_size
        W_out = output_size
        output_size = [H_out, W_out]
    else:
        H_out, W_out = output_size

    if output_size[0] is None:
        output_size[0] = H
        H_out = H
    if output_size[1] is None:
        output_size[1] = W
        W_out = W

    out = np.zeros((N, C, H_out, W_out)) if data_format == "NCHW" else np.zeros((N, H_out, W_out, C))

    for i in range(H_out):
        in_h_start = adaptive_start_index(i, H, output_size[0])
        in_h_end = adaptive_end_index(i, H, output_size[0])

        for j in range(W_out):
            in_w_start = adaptive_start_index(j, W, output_size[1])
            in_w_end = adaptive_end_index(j, W, output_size[1])

            if data_format == "NCHW":
                x_masked = x[:, :, in_h_start:in_h_end, in_w_start:in_w_end]
                if pool_type == "avg":
                    field_size = (in_h_end - in_h_start) * (in_w_end - in_w_start)
                    out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / field_size
                elif pool_type == "max":
                    out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
            elif data_format == "NHWC":
                x_masked = x[:, in_h_start:in_h_end, in_w_start:in_w_end, :]
                if pool_type == "avg":
                    field_size = (in_h_end - in_h_start) * (in_w_end - in_w_start)
                    out[:, i, j, :] = np.sum(x_masked, axis=(1, 2)) / field_size
                elif pool_type == "max":
                    out[:, i, j, :] = np.max(x_masked, axis=(1, 2))
    return out


@pytest.mark.api_nn_AdaptiveAvgPool2D_vartype
def test_adaptive_avg_pood2D_base():
    """
    base
    """
    x = randtool("int", -10, 10, [2, 3, 4, 4])
    output_size = [3, 3]
    res = compute_adaptive_pool2D(x=x, output_size=output_size)
    obj.base(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    output_size = [1, 1]
    res = compute_adaptive_pool2D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D1():
    """
    output_size = [2, 3]
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    output_size = [2, 3]
    res = compute_adaptive_pool2D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D2():
    """
    output_size = [1, 4]
    """
    x = randtool("float", -10, 10, [2, 3, 4, 4])
    output_size = [1, 4]
    res = compute_adaptive_pool2D(x=x, output_size=output_size)
    obj.run(res=res, data=x, output_size=output_size)


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D3():
    """
    output_size = [3, 3]
    """
    x = randtool("float", -10, 10, [2, 4, 4, 3])
    output_size = [3, 3]
    res = compute_adaptive_pool2D(x=x, output_size=output_size, data_format="NHWC")
    obj.run(res=res, data=x, output_size=output_size, data_format="NHWC")


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D4():
    """
    output_size = int(3)
    """
    x = randtool("float", -10, 10, [2, 4, 4, 3])
    output_size = 3
    res = compute_adaptive_pool2D(x=x, output_size=output_size, data_format="NHWC")
    obj.run(res=res, data=x, output_size=output_size, data_format="NHWC")


@pytest.mark.api_nn_AdaptiveAvgPool2D_parameters
def test_adaptive_avg_pood2D5():
    """
    output_size = tuple(3, 3)
    """
    x = randtool("float", -10, 10, [2, 4, 4, 3])
    output_size = (3, 3)
    res = compute_adaptive_pool2D(x=x, output_size=output_size, data_format="NHWC")
    obj.run(res=res, data=x, output_size=output_size, data_format="NHWC")


@pytest.mark.api_nn_AdaptiveAvgPool2D_exception
def test_adaptive_avg_pood2D6():
    """
    exception data_format = wrong
    """
    x = randtool("float", -10, 10, [2, 4, 4, 3])
    output_size = (3, 3)
    compute_adaptive_pool2D(x=x, output_size=output_size, data_format="wrong")
    obj.exception(etype=ValueError, mode="python", data=x, output_size=output_size, data_format="wrong")
