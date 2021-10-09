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


def upsample_2d(img, type_="float64", scale_factor=None, size=None, data_format="NCHW"):
    """
    def
    """
    h_in = img.shape[0]
    w_in = img.shape[1]
    if size is None:
        size = [h_in * scale_factor, w_in * scale_factor]
    global emptyImage
    h_out = size[0]
    w_out = size[1]
    if data_format == "NCHW":
        num_batchs, channels, height, width = img.shape
        emptyImage = np.zeros((num_batchs, channels, h_out, w_out), type_)
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
        emptyImage = np.zeros((num_batchs, channels, h_out, w_out), type)
        sh = h_out / height
        sw = w_out / width
        for i in range(h_out):
            for j in range(w_out):
                x = int(i / sh)
                y = int(j / sw)
                emptyImage[:, :, i, j] = img[:, :, x, y]
        emptyImage = emptyImage.transpose((0, 2, 3, 1))
    return emptyImage


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


obj = TestUpsamplingNearest2d(paddle.nn.UpSamplingNearest2D)


@pytest.mark.api_nn_UpsamplingNearest2d_vartype
def test_upsamplingnearest2d_base():
    """
    base
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [12, 12]
    res = upsample_2d(x, size=size)
    obj.base(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_vartype
def test_upsamplingnearest2d():
    """
    default
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [12, 12]
    res = upsample_2d(x, size=size)
    obj.run(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d1():
    """
    data_format = "NCHW"
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = [12, 12]
    res = upsample_2d(x, size=size, data_format=data_format)
    obj.run(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d2():
    """
    data_format = "NHWC"
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NHWC"
    size = [12, 12]
    res = upsample_2d(x, size=size, data_format=data_format)
    obj.run(res=res, data=x, size=size, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d1():
    """
    size = None, scale_factor = 2
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = 2
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)


@pytest.mark.api_nn_UpsamplingNearest2d_parameters
def test_upsamplingnearest2d1():
    """
    size = None, scale_factor = 5
    """
    x = randtool("float", -10, 10, [2, 3, 6, 10])
    data_format = "NCHW"
    size = None
    scale_factor = 5
    res = upsample_2d(x, size=size, scale_factor=scale_factor, data_format=data_format)
    obj.run(res=res, data=x, size=size, scale_factor=scale_factor, data_format=data_format)
