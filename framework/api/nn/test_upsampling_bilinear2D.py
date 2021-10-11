#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_upsampling_bilinear2D
"""

from apibase import randtool, compare
import paddle
import pytest
import numpy as np


def numpy_upsampling_bilinear2D(x, size=None, scale_factor=None, data_format="NCHW"):
    """
    implement bilinear by numpy.
    """
    if data_format == "NHWC":
        x = np.transpose(x, [0, 3, 1, 2])
    n, c, h, w = x.shape
    if size is None:
        new_size = [int(scale_factor * h), int(scale_factor * w)]
    elif isinstance(size, int):
        new_size = [size, size]
    elif isinstance(size, tuple) or isinstance(size, list):
        new_size = [size[0], size[1]]
    scale_factor = [(h - 1) / (new_size[0] - 1), (w - 1) / (new_size[1] - 1)]
    h_index = np.repeat(np.arange(new_size[0]), new_size[1]).reshape(*new_size)
    w_index = np.tile(np.arange(new_size[1]), [new_size[0]]).reshape(*new_size)

    # h_index_c = (h_index + 0.5) * scale_factor[0] - 0.5
    # w_index_c = (w_index + 0.5) * scale_factor[1] - 0.5
    h_index_c = h_index * scale_factor[0]
    w_index_c = w_index * scale_factor[1]
    h_index_c[h_index_c < 0] = 0.0
    w_index_c[w_index_c < 0] = 0.0

    h_index_c_down = h_index_c.astype(int)
    w_index_c_down = w_index_c.astype(int)
    h_index_c_up, w_index_c_up = h_index_c_down + 1, w_index_c_down + 1
    h_max, w_max = h - 1, w - 1
    h_index_c_up[h_index_c_up > h_max] = h_max
    w_index_c_up[w_index_c_up > w_max] = w_max

    top_left = x[:, :, h_index_c_down, w_index_c_down]
    bot_left = x[:, :, h_index_c_up, w_index_c_down]
    bot_right = x[:, :, h_index_c_up, w_index_c_up]
    top_right = x[:, :, h_index_c_down, w_index_c_up]

    m, n = np.modf(h_index_c)[0], np.modf(w_index_c)[0]
    res = top_left * (1 - m) * (1 - n) + bot_left * m * (1 - n) + top_right * (1 - m) * n + bot_right * m * n
    if data_format == "NHWC":
        res = np.transpose(res, [0, 2, 3, 1])
    return res


@pytest.mark.api_nn_UpsamplingBilinear2D_vartype
def test_upsampling_bilinear2D_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm1():
    """
    size: [224, 224] -> [448, 448]
    """
    x = randtool("float", -10, 10, [2, 3, 224, 224])
    size = [448, 448]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm2():
    """
    size: [224, 224] -> [112, 112]
    """
    x = randtool("float", -10, 10, [2, 3, 224, 224])
    size = [112, 112]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm3():
    """
    scale_facor: 3.75
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = 3.75
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm4():
    """
    scale_facor: 0.75
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = 0.75
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm4():
    """
    scale_facor: 0.75
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = 0.75
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm5():
    """
    data_format
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NHWC"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm6():
    """
    name
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(
        size=size, scale_factor=scale_factor, data_format=data_format, name="test"
    )(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm6():
    """
    value
    """
    x = randtool("float", -255, 255, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)
