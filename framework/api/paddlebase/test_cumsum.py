#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
test cumsum
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestCumsum(APIBase):
    """
    test cumsum
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestCumsum(paddle.cumsum)


@pytest.mark.api_base_cumsum_vartype
def test_cumsum_base():
    """
    cumsum_1D_base
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    res = np.cumsum(x_data)
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_cumsum_parameters
def test_cumsum_1D_tensor():
    """
    cumsum_1D_tensor
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    res = np.cumsum(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_cumsum_parameters
def test_cumsum_axis_1():
    """
    cumsum_axis_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.cumsum(x_data, axis=-4)
    obj.run(res=res, x=x_data, axis=-4)


@pytest.mark.api_base_cumsum_parameters
def test_cumsum_axis_2():
    """
    cumsum_axis_2
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.cumsum(x_data, axis=3)
    obj.run(res=res, x=x_data, axis=3)


'''
def test_cumsum_broadcast_2D_tensor():
    cumsum_2D_tensor
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.float32)
    res = np.cumsum(x_data)
    obj.run(res=res, x=x_data)
'''


@pytest.mark.api_base_cumsum_parameters
def test_cumsum_3():
    """
    large tensor
    """
    x_data = np.random.rand(1, 16, 96, 32).astype(np.float32)
    res = np.cumsum(x_data, axis=2)
    obj.enable_backward = False
    obj.run(res=res, x=x_data, axis=2)
