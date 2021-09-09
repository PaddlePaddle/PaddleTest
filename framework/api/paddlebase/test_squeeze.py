#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_squeeze
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSqueeze(APIBase):
    """
    test squeeze
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int8, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestSqueeze(paddle.squeeze)


@pytest.mark.api_base_squeeze_vartype
def test_squeeze_base():
    """
    squeeze_base
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_squeeze_parameters
def test_squeeze_axis1():
    """
    axis = None
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_squeeze_parameters
def test_squeeze_axis2():
    """
    axis = -4
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])
    obj.run(res=res, x=x_data, axis=-4)


@pytest.mark.api_base_squeeze_parameters
def test_squeeze_axis3():
    """
    axis = [2,3]
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])
    obj.run(res=res, x=x_data, axis=[2, 3])


@pytest.mark.api_base_squeeze_parameters
def test_squeeze_axis4():
    """
    axis = (-4,-3)
    """
    x_data = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    res = np.array([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])
    obj.run(res=res, x=x_data, axis=(-4, -3))
