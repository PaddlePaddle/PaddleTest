#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
test paddle.subtract
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSubtract(APIBase):
    """
    test subtract
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestSubtract(paddle.subtract)


@pytest.mark.api_base_subtract_vartype
def test_subtract_base():
    """
    x_data = np.array([[1, 2], [7, 8]])
    y_data = np.array([[5, 6], [3, 4]])
    """
    x_data = np.array([[1, 2], [7, 8]])
    y_data = np.array([[5, 6], [3, 4]])
    res = np.array([[-4, -4], [4, 4]])
    obj.base(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_subtract_parameters
def test_subtract1():
    """
    x_data = np.array([2, np.nan, 5]).astype(np.float32)
    y_data = np.array([1, 4, np.nan]).astype(np.float32)
    """
    x_data = np.array([2, np.nan, 5]).astype(np.float32)
    y_data = np.array([1, 4, np.nan]).astype(np.float32)
    res = np.array([1.0, np.nan, np.nan])
    obj.enable_backward = False
    obj.run(res=res, x=x_data, y=y_data)
    obj.enable_backward = True


@pytest.mark.api_base_subtract_parameters
def test_subtract2():
    """
    x_data = np.array([5, np.inf, -np.inf]).astype(np.float32)
    y_data = np.array([1, 4, 5]).astype(np.float32)
    """
    x_data = np.array([5, np.inf, -np.inf]).astype(np.float32)
    y_data = np.array([1, 4, 5]).astype(np.float32)
    res = np.array([4.0, np.inf, -np.inf])
    obj.enable_backward = False
    obj.run(res=res, x=x_data, y=y_data)
    obj.enable_backward = True
