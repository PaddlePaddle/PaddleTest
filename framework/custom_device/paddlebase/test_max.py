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
        self.no_grad_var = ["axis"]


obj = TestMax(paddle.max)


@pytest.mark.api_base_max_vartype
def test_max_base():
    """
    max_base
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data)
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_max_parameters
def test_max_2D_tensor():
    """
    max_2D_tensor
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data)
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
def test_max_axis_3():
    """
    axis = Tensor(1)
    """
    paddle.disable_static()
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    res = np.max(x_data, axis=1)
    exp = paddle.max(paddle.to_tensor(x_data), axis=paddle.to_tensor(1))
    assert np.allclose(exp.numpy(), res)


@pytest.mark.api_base_max_parameters
def test_max_axis_4():
    """
    axis = Tensor([1])
    """
    x_data = np.arange(6).reshape(2, 3).astype(np.float32)
    axis = np.array([1])
    res = np.max(x_data, axis=1)
    obj.static = False
    obj.run(res=res, x=x_data, axis=axis)
    obj.static = True


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
    res = np.max(x_data)
    np.array(-0.20009832)
    obj.run(res=res, x=x_data, axis=[-2, 1], keepdim=False)
