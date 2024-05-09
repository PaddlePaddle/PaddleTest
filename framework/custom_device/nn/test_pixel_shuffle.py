#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_pixel_shuffle
"""

from apibase import APIBase
from apibase import randtool, compare
import paddle
import pytest
import numpy as np


def pixel_shuffle_np(x, up_factor, data_format="NCHW"):
    """
    pixel shuffle implemented by numpy.
    """
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, c // (up_factor * up_factor), up_factor, up_factor, h, w)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c // (up_factor * up_factor), h * up_factor, w * up_factor]
        npreslut = np.reshape(npresult, oshape)
        return npreslut
    else:
        n, h, w, c = x.shape
        new_shape = (n, h, w, c // (up_factor * up_factor), up_factor, up_factor)
        npresult = np.reshape(x, new_shape)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, h * up_factor, w * up_factor, c // (up_factor * up_factor)]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestPixelShuffle(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]


obj = TestPixelShuffle(paddle.nn.PixelShuffle)


@pytest.mark.api_nn_PixelShuffle_vartype
def test_pixel_shuffle_base():
    """
    Test base.

    Test base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Expected Results:
        The output of pixel shuffle implemented by numpy and paddle should be equal.
    """
    x = randtool("float", -10, 10, [2, 9, 4, 4])
    up_factor = 3
    data_format = "NCHW"
    res = pixel_shuffle_np(x, up_factor, data_format=data_format)
    obj.run(res=res, data=x, upscale_factor=up_factor, data_format=data_format)


@pytest.mark.api_nn_PixelShuffle_parameters
def test_pixel_shuffle_norm1():
    """
    Test pixel shuffle when input shape changes.

    Test Base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Changes:
        input shape: [2, 9, 4, 4] -> [4, 81, 4, 4]

    Expected Results:
        The output of pixel shuffle implemented by numpy and paddle should be equal.
    """
    x = randtool("float", -10, 10, [4, 81, 4, 4])
    up_factor = 3
    data_format = "NCHW"
    res = pixel_shuffle_np(x, up_factor, data_format=data_format)
    obj.run(res=res, data=x, upscale_factor=up_factor, data_format=data_format)


@pytest.mark.api_nn_PixelShuffle_parameters
def test_pixel_shuffle_norm2():
    """
    Test pixel shuffle when data_format changes.

    Test Base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Changes:
        input shape: [2, 9, 4, 4] -> [2, 4, 4, 9]
        data_format: 'NCHW' -> 'NHWC'

    Expected Results:
        The output of pixel shuffle implemented by numpy and paddle should be equal.
    """
    x = randtool("float", -10, 10, [2, 4, 4, 9])
    up_factor = 3
    data_format = "NHWC"
    res = pixel_shuffle_np(x, up_factor, data_format=data_format)
    obj.run(res=res, data=x, upscale_factor=up_factor, data_format=data_format)


@pytest.mark.api_nn_PixelShuffle_parameters
def test_pixel_shuffle_norm3():
    """
    Test pixel shuffle when input data channels cann't be factorized by upscale_factor.

    Test Base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Changes:
        up_factor: 3 -> 4

    Expected Results:
        when input data channels cann't be factorized by upscale_factor, raise ValueError.
    """
    x = paddle.rand(shape=[2, 9, 4, 4])
    up_factor = 4
    data_format = "NCHW"
    obj.exception(ValueError, mode="python", data=x, upscale_factor=up_factor, data_format=data_format)


@pytest.mark.api_nn_PixelShuffle_parameters
def test_pixel_shuffle_norm4():
    """
    Test pixel shuffle when input data dtype changes.

    Test Base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Changes:
        input data dtype: float -> int

    Expected Results:
        The output of pixel shuffle implemented by numpy and paddle should be equal.
    """
    x = randtool("int", -10, 10, [4, 9, 4, 4])
    up_factor = 3
    data_format = "NCHW"
    res = pixel_shuffle_np(x, up_factor, data_format=data_format)
    obj.run(res=res, data=x, upscale_factor=up_factor, data_format=data_format)


@pytest.mark.api_nn_PixelShuffle_parameters
def test_pixel_shuffle_norm5():
    """
    Test pixel shuffle when input value range changes.

    Test Base config:
        x = randtool("float", -10, 10, [2, 9, 4, 4])
        up_factor = 3
        data_format = "NCHW"

    Changes:
        input value: range(-10, 10) -> range(-2555, 2555)

    Expected Results:
        The output of pixel shuffle implemented by numpy and paddle should be equal.
    """
    x = randtool("float", -2555, 2555, [2, 9, 4, 4])
    up_factor = 3
    data_format = "NCHW"
    res = pixel_shuffle_np(x, up_factor, data_format=data_format)
    obj.run(res=res, data=x, upscale_factor=up_factor, data_format=data_format)
