#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_upsamplingbilinear2d
"""
from apibase import APIBase
from apibase import randtool
import pytest
import paddle
import paddle.fluid as fluid
import numpy as np


class TestUpsamplingBilinear2d(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True
        self.no_grad_var = ["scale_factor", "size"]


obj = TestUpsamplingBilinear2d(paddle.nn.UpsamplingBilinear2D)


def upsample_2d(img, scale_factor=None, size=[12, 12], data_format="NCHW"):
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
        if (
            str(type(scale_factor)) == "<class 'tuple'>"
            or str(type(scale_factor)) == "<class 'list'>"
            or str(type(scale_factor)) == "<class 'numpy.ndarray'>)"
        ):
            size = [int(h_in * scale_factor[0]), int(w_in * scale_factor[1])]
        else:
            size = [int(h_in * scale_factor), int(w_in * scale_factor)]
    dsth = size[0]
    dstw = size[1]
    if data_format == "NCHW":
        num_batches, channels, height, width = img.shape
        emptyImage = np.zeros((num_batches, channels, size[0], size[1]))
        srch = size[0] / height
        srcw = size[1] / width
        for num_batche in range(num_batches):
            for channel in range(channels):
                for i in range(dsth):
                    for j in range(dstw):
                        srcY = i * (srch / dsth)
                        srcX = j * (srcw / dstw)
                        # srcY = (i+0.5) * (srch / dsth) -0.5
                        # srcX = (j+0.5) * (srcw / dstw) -0.5
                        x1, x2 = int(np.floor(srcX)), int(srcX) + 1
                        y1, y2 = int(np.floor(srcY)), int(srcY) + 1
                        x2 = min(x1 + 1, srcw - 1)
                        y2 = min(y1 + 1, srch - 1)
                        # fl.write("{} {} {} {}\n".format(x1,x2,y1,y2))
                        # print(j,y1,y2,srcY)
                        f1 = (x2 - srcX) * img[num_batche, channel, y1, x2] + (srcX - x1) * img[
                            num_batche, channel, y1, x1
                        ]
                        f2 = (x2 - srcX) * img[num_batche, channel, y2, x2] + (srcX - x1) * img[
                            num_batche, channel, y2, x2
                        ]
                        f = (srcY - y1) * f1 + (y2 - srcY) * f2
        emptyImage[num_batche, channel, i, j] = f
    elif data_format == "NHWC":
        img = img.transpose((0, 3, 1, 2))
        num_batches, channels, height, width = img.shape
        emptyImage = np.zeros((num_batches, channels, size[0], size[1]))
        srch = size[0] / height
        srcw = size[1] / width
        for num_batche in range(num_batches):
            for channel in range(channels):
                for i in range(dsth):
                    for j in range(dstw):
                        srcY = i * (srch / dsth)
                        srcX = j * (srcw / dstw)
                        # srcY = (i+0.5) * (srch / dsth) -0.5
                        # srcX = (j+0.5) * (srcw / dstw) -0.5
                        x1, x2 = int(np.floor(srcX)), int(srcX) + 1
                        y1, y2 = int(np.floor(srcY)), int(srcY) + 1
                        x2 = min(x1 + 1, srcw - 1)
                        y2 = min(y1 + 1, srch - 1)
                        # fl.write("{} {} {} {}\n".format(x1,x2,y1,y2))
                        # print(j,y1,y2,srcY)
                        f1 = (x2 - srcX) * img[num_batche, channel, y1, x2] + (srcX - x1) * img[
                            num_batche, channel, y1, x1
                        ]
                        f2 = (x2 - srcX) * img[num_batche, channel, y2, x2] + (srcX - x1) * img[
                            num_batche, channel, y2, x2
                        ]
                        f = (srcY - y1) * f1 + (y2 - srcY) * f2
        emptyImage[num_batche, channel, i, j] = f
        emptyImage = emptyImage.transpose((0, 2, 3, 1))
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
    scale_factor = (2, 3)
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d6():
    """
    test when size is a tuple like (12,13)
    expected results:
    the results of numpy and paddle api should not be different.
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = (12, 13)
    scale_factor = None
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d7():
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
