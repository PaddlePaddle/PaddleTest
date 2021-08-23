#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test strided_slice
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestStridedSlice(APIBase):
    """
    test strided_slice
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestStridedSlice(paddle.strided_slice)
obj1 = TestStridedSlice(paddle.strided_slice)
obj1.types = [np.int32, np.int64]


@pytest.mark.api_base_strided_slice_vartype
def test_strided_slice_base():
    """
    base float
    """
    x = randtool("float", -10, 10, (3, 4, 5, 6))
    axes = [1, 2, 3]
    starts = [-3, 0, 2]
    ends = [3, 2, 4]
    strides = [1, 1, 1]
    tmp = x
    res = tmp[:, -3:3:1, 0:2:1, 2:4:1]
    obj.base(res=res, x=x, axes=axes, starts=starts, ends=ends, strides=strides)


@pytest.mark.api_base_strided_slice_vartype
def test_strided_slice_base1():
    """
    base int
    """
    x = randtool("int", -10, 10, (3, 4, 5, 6))
    axes = [1, 2, 3]
    starts = [-3, 0, 2]
    ends = [3, 2, 4]
    strides = [1, 1, 1]
    tmp = x
    res = tmp[:, -3:3:1, 0:2:1, 2:4:1]
    obj1.base(res=res, x=x, axes=axes, starts=starts, ends=ends, strides=strides)


@pytest.mark.api_base_strided_slice_parameters
def test_strided_slice_1():
    """
    float
    x.shape=[5, 8, 6, 4, 2, 6]
    axes=[1, 2, 5]
    starts=[-3, 0, 2]
    ends=[3, 2, 4]
    strides=[-1, -1, -2]
    """
    x = randtool("float", -10, 10, (5, 8, 6, 4, 2, 6))
    axes = [1, 2, 5]
    starts = [-3, 3, 4]
    ends = [3, 0, 1]
    strides = [-1, -1, -2]
    tmp = x
    res = tmp[:, -3:3:-1, 3:0:-1, :, :, 4:1:-2]
    obj.run(res=res, x=x, axes=axes, starts=starts, ends=ends, strides=strides)


@pytest.mark.api_base_strided_slice_parameters
def test_strided_slice_2():
    """
    int
    x.shape=[5, 8, 6, 4, 2, 6]
    axes = [1, 2, 5]
    starts = [6, 5, 4]
    ends = [2, 0, 1]
    strides = [-1, -2, -3]
    """
    x = randtool("int", -10, 10, (5, 8, 6, 4, 2, 6))
    axes = [1, 2, 5]
    starts = [6, 5, 4]
    ends = [2, 0, 1]
    strides = [-1, -2, -3]
    tmp = x
    res = tmp[:, 6:2:-1, 5:0:-2, :, :, 4:1:-3]
    obj.run(res=res, x=x, axes=axes, starts=starts, ends=ends, strides=strides)
