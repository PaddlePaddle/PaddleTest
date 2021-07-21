#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test less equal
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLessEqual(APIBase):
    """
    test less_equal
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


obj = TestLessEqual(paddle.less_equal)


@pytest.mark.api_base_less_equal_vartype
def test_less_equal_base():
    """
    less_equal_base
    """
    x_data = np.array([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]])
    y_data = np.array([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]])
    res = np.less_equal(x_data, y_data)
    obj.base(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_less_equal_parameters
def test_less_equal_dicimal():
    """
    less_equal_dicimal
    """
    x_data = np.array([[2.0, 1.0, -3.5], [-2.7, 1.5, 3], [0, 4.1, 8.6]])
    y_data = np.array([[-2.0, 1.1, -3.5], [-2.5, 1.5, 3.5], [0.5, 4.2, 8.3]])
    res = np.less_equal(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_less_equal_parameters
def test_less_equal_1D_tensor():
    """
    1D_tensor
    """
    x_data = np.array([1]).astype(np.float32)
    y_data = np.array([1, -1, 2, -4]).astype(np.float32)
    res = np.less_equal(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_less_equal_parameters
def test_less_equal_broadcast_1():
    """
    broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.float32)
    res = np.less_equal(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_less_equal_parameters
def test_less_equal_broadcast_2():
    """
    broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.float32)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.float32)
    res = np.less_equal(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
