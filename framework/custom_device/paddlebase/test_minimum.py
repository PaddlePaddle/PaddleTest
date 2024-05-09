#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test minimum
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMinimum(APIBase):
    """
    test minimum
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestMinimum(paddle.minimum)


@pytest.mark.api_base_minimum_vartype
def test_minimum_1D_tensor():
    """
    minimum_1D_tensor
    """
    x_data = np.array([1]).astype(np.float32)
    y_data = np.array([2, -2, 0, 3]).astype(np.float32)
    res = np.minimum(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_minimum_parameters
def test_minimum_broadcast_1():
    """
    minimum_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.float32)
    res = np.minimum(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_minimum_parameters
def test_minimum_broadcast_2():
    """
    minimum_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.float32)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.float32)
    res = np.minimum(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
