#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test min
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestMin(APIBase):
    """
    test min
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestMin(paddle.min)


@pytest.mark.api_base_min_vartype
def test_min_base():
    """
    min_base
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.array([0])
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_min_parameters
def test_min_2D_tensor():
    """
    min_2D_tensor
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.array([0])
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_min_parameters
def test_min_axis_1():
    """
    axis = -2
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.min(x_data, axis=-2)
    obj.run(res=res, x=x_data, axis=-2)


@pytest.mark.api_base_min_parameters
def test_min_axis_2():
    """
    axis =1
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.min(x_data, axis=1)
    obj.run(res=res, x=x_data, axis=1)


@pytest.mark.api_base_min_parameters
def test_min_2D_keepdim():
    """
    keepdim=True
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.min(x_data, axis=0, keepdims=True)
    obj.run(res=res, x=x_data, axis=0, keepdim=True)
