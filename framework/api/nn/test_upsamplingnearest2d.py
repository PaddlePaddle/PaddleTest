#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_upsamplingnearest2d
"""
from apibase import APIBase
from apibase import randtool
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np


class TestUpsamplingNearest2d(APIBase):
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


obj = TestUpsamplingNearest2d(paddle.nn.UpsamplingNearest2D)


def upsample_2d(img, scale_factor=None, size=None, data_format="NCHW"):
    """
    def
    """
    if data_format == "NCHW":
        h_in = img.shape[2]
        w_in = img.shape[3]
    else:
        h_in = img.shape[1]
        w_in = img.shape[2]
    if size is None:
        if str(type(scale_factor)) == "<class 'list'>" or str(type(scale_factor)) == "<class 'paddle.Tensor'>":
            size = [h_in * scale_factor[0], w_in * scale_factor[1]]
        else:
            size = [h_in * scale_factor, w_in * scale_factor]
    h_out = size[0]
    w_out = size[1]
    if data_format == "NCHW":
        num_batchs, channels, height, width = img.shape
        emptyImage = np.zeros((num_batchs, channels, h_out, w_out))
        sh = h_out / height
        sw = w_out / width
        for i in range(h_out):
            for j in range(w_out):
                x = int(i / sh)
                y = int(j / sw)
                emptyImage[:, :, i, j] = img[:, :, x, y]
    elif data_format == "NHWC":
        img = img.transpose((0, 3, 1, 2))
        num_batchs, channels, height, width = img.shape
        print(num_batchs, channels, height, width)
        emptyImage = np.zeros((num_batchs, channels, h_out, w_out))
        sh = h_out / height
        sw = w_out / width
        for i in range(h_out):
            for j in range(w_out):
                x = int(i / sh)
                y = int(j / sw)
                emptyImage[:, :, i, j] = img[:, :, x, y]
    return emptyImage


@pytest.mark.api_nn_UpsamplingNearest2d_vartype
def test_upsamplingnearest2d_base():
    """
    base

    Test the base config of upsamplingnearest2d API

    input:x with shape [2,3,6,10] and data_type float
    data_format="NCHW"
    size=[12,12]

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [12, 12]
    res = upsample_2d(x, size=size, data_format=data_format)
    obj.base(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d():
    """
    Test when size is larger than base case.

    size = [256,256]

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [256, 256]
    res = upsample_2d(x, size=size, data_format=data_format)
    obj.run(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d1():
    """
    test when data_format = "NHWC"

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NHWC"
    size = [12, 12]
    res = upsample_2d(x, size=size, data_format=data_format)
    obj.run(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d2():
    """
    test when size = None, scale_factor = 2
    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = 2
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d3():
    """
    test when size = None, scale_factor = 5,scale_factor is larger

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = 5
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d4():
    """
    test when scale_factor is a list like [2,3]

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = [2, 3]
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d5():
    """
    test when scale_factor is a tuple like (2,3)

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = [2, 3]
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d6():
    """
    test when scale_factor is a tensor like Tensor([2,3])

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = paddle.to_tensor([2, 3])
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d7():
    """
    test when size is a tuple like (12,13)

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [12, 13]
    scale_factor = None
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d8():
    """
    test when size is a Tensor like Tensor([12,13])

    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = paddle.to_tensor([12, 13])
    scale_factor = None
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d9():
    """
    test when input x contains values less than 0.

    expected results :
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    x[x >= 0] = x[x >= 0] * -1
    data_format = "NCHW"
    size = [12, 13]
    scale_factor = None
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)
