#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
*
* Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
* @file test_avg_pool3D.py
* @author zhengtianyu
* @date 2020-08-28 14:05:15
* @brief test_avg_pool3D
*
**************************************************************************/
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAvgPool3D(APIBase):
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


obj = TestAvgPool3D(paddle.nn.AvgPool3D)


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


def pool3D_forward_naive(x,
                         ksize,
                         strides,
                         paddings,
                         global_pool=0,
                         ceil_mode=False,
                         exclusive=True,
                         adaptive=False,
                         data_format='NCDHW',
                         pool_type='max',
                         padding_algorithm="EXPLICIT"):
    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                             "It can only be 'SAME' or 'VALID'." %
                             str(padding_algorithm))

        if padding_algorithm == "VALID":
            paddings = [0, 0, 0, 0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode)"
                    " must be False. "
                    "Received ceil_mode: True.")
        elif padding_algorithm == "SAME":
            input_data_shape = []
            if data_format == "NCDHW":
                input_data_shape = x.shape[2:5]
            elif data_format == "NDHWC":
                input_data_shape = x.shape[1:4]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    assert len(paddings) == 3 or len(paddings) == 6
    is_sys = True if len(paddings) == 3 else False

    N = x.shape[0]
    C, D, H, W = [x.shape[1], x.shape[2], x.shape[3], x.shape[4]] \
        if data_format == 'NCDHW' else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]

    if global_pool == 1:
        ksize = [D, H, W]
        paddings = [0 for _ in range(len(paddings))]

    pad_d_forth = paddings[0] if is_sys else paddings[0]
    pad_d_back = paddings[0] if is_sys else paddings[1]
    pad_h_up = paddings[1] if is_sys else paddings[2]
    pad_h_down = paddings[1] if is_sys else paddings[3]
    pad_w_left = paddings[2] if is_sys else paddings[4]
    pad_w_right = paddings[2] if is_sys else paddings[5]

    if adaptive:
        D_out, H_out, W_out = ksize
    else:

        D_out = (D - ksize[0] + pad_d_forth + pad_d_back + strides[0] - 1) // strides[0] + 1 \
            if ceil_mode else (D - ksize[0] + pad_d_forth + pad_d_back) // strides[0] + 1

        H_out = (H - ksize[1] + pad_h_up + pad_h_down + strides[1] - 1) // strides[1] + 1 \
            if ceil_mode else (H - ksize[1] + pad_h_up + pad_h_down) // strides[1] + 1

        W_out = (W - ksize[2] + pad_w_left + pad_w_right + strides[2] - 1) // strides[2] + 1 \
            if ceil_mode else (W - ksize[2] + pad_w_left + pad_w_right) // strides[2] + 1


    out = np.zeros((N, C, D_out, H_out, W_out)) if data_format == 'NCDHW' \
        else np.zeros((N, D_out, H_out, W_out, C))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])

        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])

            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                else:

                    d_start = k * strides[0] - pad_d_forth
                    d_end = np.min((k * strides[0] + ksize[0] - pad_d_forth,
                                    D + pad_d_back))
                    h_start = i * strides[1] - pad_h_up
                    h_end = np.min(
                        (i * strides[1] + ksize[1] - pad_h_up, H + pad_h_down))
                    w_start = j * strides[2] - pad_w_left
                    w_end = np.min((j * strides[2] + ksize[2] - pad_w_left,
                                    W + pad_w_right))

                    field_size = (d_end - d_start) * (h_end - h_start) * (
                        w_end - w_start)
                    w_start = np.max((w_start, 0))
                    d_start = np.max((d_start, 0))
                    h_start = np.max((h_start, 0))
                    w_end = np.min((w_end, W))
                    d_end = np.min((d_end, D))
                    h_end = np.min((h_end, H))
                if data_format == 'NCDHW':
                    x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:
                                 w_end]
                    if pool_type == 'avg':
                        if (exclusive or adaptive):
                            field_size = (d_end - d_start) * (
                                h_end - h_start) * (w_end - w_start)

                        out[:, :, k, i, j] = np.sum(x_masked,
                                                    axis=(2, 3, 4)) / field_size
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
                    x_masked = x[:, d_start:d_end, h_start:h_end, w_start:
                                 w_end, :]
                    if pool_type == 'avg':
                        if (exclusive or adaptive):
                            field_size = (d_end - d_start) * (
                                h_end - h_start) * (w_end - w_start)

                        out[:, k, i, j, :] = np.sum(x_masked,
                                                    axis=(1, 2, 3)) / field_size
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))

    return out


def avg_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             data_format='NCDHW',
                             adaptive=False):
    """
    avg_pool3D_forward_naive
    """
    out = pool3D_forward_naive(
        x=x,
        ksize=ksize,
        strides=strides,
        paddings=paddings,
        global_pool=global_pool,
        ceil_mode=ceil_mode,
        exclusive=exclusive,
        adaptive=adaptive,
        data_format=data_format,
        pool_type="avg")
    return out


@pytest.mark.p0
def test_avg_pool3D_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.base(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=False)


def test_avg_pool3D():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=False)


def test_avg_pool3D1():
    """
    ceil_mode = True BUG!  GPU Kernel
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    ceil_mode = True
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding, ceil_mode=ceil_mode,
     exclusive=False)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)


def test_avg_pool3D2():
    """
    data_format = 'NDHWC'
    """
    x = randtool("float", -10, 10, [2, 8, 8, 8, 3])
    kernel_size = [3, 3, 3]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    data_format = 'NDHWC'
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding, data_format=data_format)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride,
    padding=padding, data_format=data_format, exclusive=False)


def test_avg_pool3D3():
    """
    stride = [1, 2, 3]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [1, 2, 3]
    padding = [0, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=False)


def test_avg_pool3D4():
    """
    stride = [3, 2, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding = [0, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=False)


def test_avg_pool3D5():
    """
    stride = [3, 2, 1], padding=[1, 0, 0]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding=[1, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D6():
    """
    stride = [3, 2, 1], padding=[1, 1, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding=[1, 1, 1]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D7():
    """
    stride = [3, 2, 1], padding=[1, 2, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding=[1, 2, 1]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D8():
    """
    padding is int
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding = 1
    paddings=[1, 1, 1]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=paddings)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D9():
    """
    stride is int
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = 2
    strides = [2, 2, 2]
    padding = 1
    paddings=[1, 1, 1]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=strides, paddings=paddings)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D10():
    """
    stride is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = (3, 2, 1)
    padding=[1, 0, 0]
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D11():
    """
    padding is tuple
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = (3, 2, 1)
    padding= (1, 0, 0)
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding, exclusive=True)


def test_avg_pool3D12():
    """
    stride = [3, 2, 1]
    """
    x = randtool("float", -10, 10, [2, 3, 8, 8, 8])
    kernel_size = [3, 3, 3]
    stride = [3, 2, 1]
    padding = [0, 0, 0]
    exclusive = False
    res = avg_pool3D_forward_naive(x=x, ksize=kernel_size, strides=stride, paddings=padding,
    exclusive=exclusive)
    obj.run(res=res, data=x, kernel_size=kernel_size, stride=stride, padding=padding,
    exclusive=exclusive)
