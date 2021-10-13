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
    if isinstance(scale_factor, (float, int)):
        scale_factor = [scale_factor, scale_factor]
    if size is None:
        new_size = [int(scale_factor[0] * h), int(scale_factor[1] * w)]
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
    Test base.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
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
    Test upsampling bilinear2d when size is tuple.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size type: list -> tuple

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = (100, 100)
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
    Test upsampling bilinear2d when size is Tensor.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size type: list -> Tensor

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    size = paddle.to_tensor(size)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm3():
    """
    Test upsampling bilinear2d when size is None and use 'scale_factor'.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size type: list -> None

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = 2.0
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
    Test upsampling bilinear2d when size is None and scale_factor increase.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> 3.75

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
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
def test_upsampling_bilinear2D_norm5():
    """
    Test upsampling bilinear2d when size is None and scale_factor decrease.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> 0.75

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
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
def test_upsampling_bilinear2D_norm6():
    """
    Test upsampling bilinear2d when size is None and scale_factor is [scale_factor_h, scale_factor_w].

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> [1, 2]

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
        The output shape should equal [2, 3, 50, 100]
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = [1, 2]
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)
    assert paddle_res.shape == (2, 3, 50, 100), "The output shape isn't expected."


def test_upsampling_bilinear2D_norm7():
    """
    Test upsampling bilinear2d when size is None and scale_factor is (scale_factor_h, scale_factor_w).

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> (1, 2)

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
        The output shape should equal [2, 3, 50, 100]
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = (1, 2)
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)
    assert paddle_res.shape == (2, 3, 50, 100), "The output shape isn't expected."


def test_upsampling_bilinear2D_norm8():
    """
    Test upsampling bilinear2d when size is None and scale_factor is Tensor([scale_factor_h, scale_factor_w]).

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> Tensor([1, 2])

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
        The output shape should equal (2, 3, 50, 100)
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = [1, 2]
    data_format = "NCHW"
    delta = 1e-3
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    scale_factor = paddle.to_tensor(scale_factor)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)
    assert paddle_res.shape == (2, 3, 50, 100), "The output shape isn't expected."


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm9():
    """
    Test upsampling bilinear2d 'size' and 'scale_factor' are both None.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = None
        scale_factor: 2.0 -> None

    Expected Results:
        One of size or scale_factor should be defined, raise ValueError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = None
    data_format = "NCHW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(ValueError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm10():
    """
    Test upsampling bilinear2d 'size' and 'scale_factor' are both setted.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size = [100, 100]
        scale_factor: 2.0

    Expected Results:
        Only one of size or scale_factor should be defined, raise ValueError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = 2.0
    data_format = "NCHW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(ValueError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm11():
    """
    Test upsampling bilinear2d when size'length is 4.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size: [100, 100] -> [100, 100, 100, 100]

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
        size length should be 2 for input 4-D tensor, raise ValueError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100, 100, 100]
    scale_factor = None
    data_format = "NCHW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(ValueError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm12():
    """
    Test upsampling bilinear2d when scale_factor should be greater than zero.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        scale_factor: 2.0 -> -2.0

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
        size length should be 2 for input 4-D tensor, raise ValueError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = None
    scale_factor = -2.0
    data_format = "NCHW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(ValueError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm13():
    """
    Test upsampling bilinear2d when size is dict.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        size type: (float|list|tuple|Tensor) -> dict

    Expected Results:
         size should be a list or tuple or Variable, raise TypeError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = {}
    scale_factor = None
    data_format = "NCHW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(TypeError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm14():
    """
    Test upsampling bilinear2d when data_format changes.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        data_format: 'NCHW' -> 'NHWC'

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
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
def test_upsampling_bilinear2D_norm15():
    """
    Test upsampling bilinear2d when data_format is 'NHCW'.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        data_format: "NCHW" -> 'NHCW'

    Expected Results:
         only `NCHW` or `NHWC` supported, raise ValueError.
    """
    x = randtool("float", -10, 10, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NHCW"

    tensor_x = paddle.to_tensor(x)
    with pytest.raises(ValueError):
        paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2D_norm16():
    """
    Test upsampling bilinear2d when set name.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        set paddle.nn.UpsamplingBilinear2D parameter name to "test"

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
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
def test_upsampling_bilinear2D_norm17():
    """
    Test upsampling bilinear2d when input value range changes.

    Test base config:
        x = randtool("float", -10, 10, [2, 3, 50, 50])
        size = [100, 100]
        scale_factor = None
        data_format = "NCHW"
        delta = 1e-3
        rtol = 1e-5

    Notes:
        The numerical error will increase when input shape and input value increase. So delta
        and rtol should be bigger.

    Changes:
        input value range: [-10, 10] -> [-255, 255]
        delta = 1e-2
        When range be larger, delta and rtol should increase.

    Expected Results:
        The error of upsampling bilinear2d implemented by numpy and paddle should be small.
    """
    x = randtool("float", -255, 255, [2, 3, 50, 50])
    size = [100, 100]
    scale_factor = None
    data_format = "NCHW"
    delta = 1e-2
    rtol = 1e-5
    res = numpy_upsampling_bilinear2D(x, size, scale_factor, data_format=data_format)

    tensor_x = paddle.to_tensor(x)
    paddle_res = paddle.nn.UpsamplingBilinear2D(size=size, scale_factor=scale_factor, data_format=data_format)(tensor_x)
    paddle_res = paddle_res.numpy()
    compare(res, paddle_res, delta, rtol)
