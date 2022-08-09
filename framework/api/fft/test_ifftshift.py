#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ifftshift
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestIfftshift(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64, np.float32]
        # self.debug = True
        self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestIfftshift(paddle.fft.ifftshift)


@pytest.mark.api_fft_ifftshift_vartype
def test_ifftshift_base():
    """
    base
    dtype: float
    """
    x = np.array([3, 1, 2, 2, 3], dtype=float)
    res = np.fft.ifftshift(x)
    obj.base(res=res, x=x)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_0():
    """
    default
    """
    x_data = np.mgrid[:2, :4, :2][1]
    res = np.fft.ifftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_1():
    """
    x: tensor - 3d
    """
    x_data = np.random.rand(4, 5, 4) * 10 - 5
    res = np.fft.ifftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_2():
    """
    x: tensor - 4d
    """
    x_data = np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.ifftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_3():
    """
    x: tensor - 4d
    dtype: complex
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.ifftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_4():
    """
    x: tensor - 4d
    dtype: complex
    axis = 3
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.ifftshift(x_data, axes=3)
    obj.run(res=res, x=x_data, axes=3)


@pytest.mark.api_fft_ifftshift_parameters
def test_ifftshift_5():
    """
    x: tensor - 4d
    dtype: complex
    axis = (0, 3)
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.ifftshift(x_data, axes=(0, 3))
    obj.run(res=res, x=x_data, axes=(0, 3))
