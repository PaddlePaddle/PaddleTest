#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test max
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSlice(APIBase):
    """
    test slice
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestSlice(paddle.strided_slice)


@pytest.mark.api_base_slice_vartype
def test_slice_base():
    """
    slice_base
    """
    x_data = np.arange(6)
    res = np.array([3])
    obj.base(res=res, x=x_data, axes=[0], starts=[3], ends=[1], strides=[-2])


@pytest.mark.api_base_slice_parameters
def test_slice_base_1():
    """
    axes > 0
    """
    x_data = np.arange(6 * 6).reshape((6, 6))
    axes = [0, 1]
    starts = [3, 4]
    ends = [5, 2]
    strides = [1, -2]
    res = x_data[starts[0] : ends[0] : strides[0], starts[1] : ends[1] : strides[1]]
    obj.run(res=res, x=x_data, axes=axes, starts=starts, ends=ends, strides=strides)


@pytest.mark.api_base_slice_parameters
def test_slice_base_2():
    """
    strides > max
    """
    x_data = np.arange(6 * 6).reshape((6, 6))
    axes = [0, 1]
    starts = [3, 4]
    ends = [5, 2]
    strides = [4, -2]
    res = x_data[starts[0] : ends[0] : strides[0], starts[1] : ends[1] : strides[1]]
    obj.run(res=res, x=x_data, axes=axes, starts=starts, ends=ends, strides=strides)


@pytest.mark.api_base_slice_parameters
def test_slice_3():
    """
    strides < 0
    end < 0
    set_value
    """
    paddle.disable_static()
    x = paddle.to_tensor([1, 2, 3, 4])
    r = np.array([4, 3, 2])
    assert np.allclose(x[7:-4:-1].numpy(), r)
    x[7:-4:-1] = -5
    res = np.array([1, -5, -5, -5])
    assert np.allclose(x.numpy(), res)
