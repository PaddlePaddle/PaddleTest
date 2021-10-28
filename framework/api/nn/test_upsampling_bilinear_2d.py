#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.UpsamplingBilinear2D
"""
import numpy as np
import pytest
import paddle
from apibase import APIBase
from apibase import randtool


class TestUpsamplingBilinear2D(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.debug = True
        # adapt self.delta to a larger value to allow more tolerance
        # when comparing two results
        self.delta = 1e-5
        self.no_grad_var = ["scale_factor", "size"]


obj = TestUpsamplingBilinear2D(paddle.nn.UpsamplingBilinear2D)


def upsampling_bilinear2D_using_numpy(x, size, scale_factor=None, data_format="NCHW"):
    """implementation of paddle.nn.UpsamplingBilinear2D using numpy
        size: None, or a list or tuple of 2 ints
        scale_factor: None, or a list or tuple of 2 floats
    """
    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    batch_size, channel, in_h, in_w = x.shape
    if size is not None:
        out_h, out_w = size
    elif scale_factor is not None:
        out_h = int(scale_factor[0] * in_h)
        out_w = int(scale_factor[1] * in_w)
    else:
        assert False, "size and scale_factor cannot both be None"
    out = np.zeros((batch_size, channel, out_h, out_w))
    ratio_h = (in_h - 1.0) / (out_h - 1.0)
    ratio_w = (in_w - 1.0) / (out_w - 1.0)

    for i in range(out_h):
        h = int(ratio_h * i)
        hid = 1 if h < in_h - 1 else 0
        h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            w = int(ratio_w * j)
            wid = 1 if w < in_w - 1 else 0
            w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda
            out[:, :, i, j] = h2lambda * (w2lambda * x[:, :, h, w] + w1lambda * x[:, :, h, w + wid]) + h1lambda * (
                w2lambda * x[:, :, h + hid, w] + w1lambda * x[:, :, h + hid, w + wid]
            )

    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(x.dtype)


@pytest.mark.api_nn_UpsamplingBilinear2D_vartype
def test_upsampling_bilinear2d_base():
    """
    test all dtypes in self.types of APIBase,
    input shape = [4, 5, 6, 7]
    output shape = [4, 5, 10, 10]
    data_format = 'NCHW'

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = None
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.base(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d():
    """
    size = None
    scale_factor = [1.2, 3.3]
    data_format = 'NCHW'
    input shape = [4, 5, 6, 7]
    output shape = [4, 5, int(6*1.2), int(7*3.3)]

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = None
    scale_factor = [1.2, 3.3]
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    assert res.shape[2] == int(6 * 1.2)
    assert res.shape[3] == int(7 * 3.3)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_2():
    """
    size = [100, 50], use larger size

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [100, 50]
    scale_factor = None
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_3():
    """
    value range of elements in x is [-1000.0, -10.0),
    because the elements in x have larger absolute values,
    so increase  obj.rtol to 1e-5 to tolerate larger difference
    when comparing results

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-1000, high=-10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = None
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    original_rtol = obj.rtol
    obj.rtol = 1e-5
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.rtol = original_rtol


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_4():
    """
    input shape = [4, 5, 20, 20]
    output shape = [4, 5, 10, 10],
    height and width of output are smaller than that of input,
    increase  obj.rtol to 1e-3 to tolerate larger difference
    when comparing results

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 20, 20])
    size = [10, 10]
    scale_factor = None
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    original_rtol = obj.rtol
    obj.rtol = 1e-3
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.rtol = original_rtol


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_5():
    """
    data_format = 'NHWC'
    input shape = [4, 5, 6, 7]
    output shape = [4, 10, 10, 7]

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = None
    data_format = "NHWC"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_6():
    """
        size = np.array(size).astype(np.int32)

        in obj.run(), because size is a numpy.ndarray, size will be
        transformed to a paddle.Tensor, and because size is in
        APIBase.no_grad_var, the 'stop_gradient' attr of size will
        be set to True to stop gradient.

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = None
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    size = np.array(size).astype(np.int32)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_parameters
def test_upsampling_bilinear2d_7():
    """
        scale_factor = np.array(scale_factor).astype(np.float32)

        in obj.run(), because scale_factor is a numpy.ndarray, scale_factor will be
        transformed to a paddle.Tensor, and because scale_factor is in
        APIBase.no_grad_var, the 'stop_gradient' attr of scale_factor will
        be set to True to stop gradient.

    expect: bilinear interpolation in numpy  is the same with
     that in paddle, so the two results should be the same within
     a tolerance
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = None
    scale_factor = [3.7, 4.1]
    data_format = "NCHW"
    res = upsampling_bilinear2D_using_numpy(x, size=size, scale_factor=scale_factor, data_format=data_format)
    scale_factor = np.array(scale_factor).astype(np.float32)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingBilinear2D_exception
def test_upsampling_bilinear2d_8():
    """
    scale_factor = [-1.1, 4.1]

    ValueError: Attr(scale) should be greater than zero.
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = None
    scale_factor = [-1.1, 4.1]
    data_format = "NCHW"
    obj.exception(
        etype=ValueError, mode="python", data=x, size=size, scale_factor=scale_factor, data_format=data_format
    )


@pytest.mark.api_nn_UpsamplingBilinear2D_exception
def test_upsampling_bilinear2d_9():
    """
    data_format = 'MCHW'

    ValueError: Got wrong value for param `data_format`: MCHW received but only `NCHW` or `NHWC` supported for 4-D input
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = None
    data_format = "MCHW"
    obj.exception(
        etype=ValueError, mode="python", data=x, size=size, scale_factor=scale_factor, data_format=data_format
    )
    
      
@pytest.mark.api_nn_UpsamplingBilinear2D_exception
def test_upsampling_bilinear2d_10():
    """
    ValueError: Only one of size or scale_factor should be defined.
    """
    x = randtool("float", low=-10, high=10, shape=[4, 5, 6, 7])
    size = [10, 10]
    scale_factor = [4.0, 5.0]
    data_format = "NCHW"
    obj.exception(
        etype=ValueError, mode="python", data=x, size=size, scale_factor=scale_factor, data_format=data_format
    )
