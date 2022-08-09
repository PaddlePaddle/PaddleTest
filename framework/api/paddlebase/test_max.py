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


class TestMax(APIBase):
    """
    test max
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


obj = TestMax(paddle.max)


@pytest.mark.api_base_max_vartype
def test_max_base():
    """
    max_base
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.array([5])
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_max_parameters
def test_max_2D_tensor():
    """
    max_2D_tensor
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.array([5])
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_max_parameters
def test_max_axis_1():
    """
    axis = -2
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data, axis=-2)
    obj.run(res=res, x=x_data, axis=-2)


@pytest.mark.api_base_max_parameters
def test_max_axis_2():
    """
    axis =1
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data, axis=1)
    obj.run(res=res, x=x_data, axis=1)


@pytest.mark.api_base_max_parameters
def test_max_2D_keepdim():
    """
    keepdim=True
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data, axis=0, keepdims=True)
    obj.run(res=res, x=x_data, axis=0, keepdim=True)


@pytest.mark.api_base_max_parameters
def test_max_1():
    """
    special input
    """
    x_data = np.array([[-1.00595951, -0.20009832], [-0.35623679, -0.95880121]])
    res = np.array([-0.20009832])
    obj.run(res=res, x=x_data, axis=[-2, 1], keepdim=False)


# x1 = paddle.to_tensor([[-1.00595951, -0.20009832], [-0.35623679, -0.95880121]])
# out = paddle.max(x1, axis=[-2, 1], keepdim=False)
# print(out)
