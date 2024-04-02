#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fftshift
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFFtshift(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64, np.float32]
        # self.debug = True
        #
        self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestFFtshift(paddle.fft.fftshift)


@pytest.mark.api_fft_fftshift_vartype
def test_fftshift_base():
    """
    base
    dtype: float
    """
    x = np.mgrid[:10].astype("float64")
    res = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    obj.base(res=res, x=x)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_0():
    """
    default
    """
    x_data = np.mgrid[:2, :4, :2][1]
    res = np.fft.fftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_1():
    """
    x: tensor-3d
    """
    x_data = np.random.rand(4, 5, 4) * 10 - 5
    res = np.fft.fftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_2():
    """
    x: tensor-4d
    """
    x_data = np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.fftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_3():
    """
    x: tensor-4d
    dtype: complex
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.fftshift(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_4():
    """
    x: tensor-4d
    dtype: complex
    axis = 3
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.fftshift(x_data, axes=3)
    obj.run(res=res, x=x_data, axes=3)


@pytest.mark.api_fft_fftshift_parameters
def test_fftshift_5():
    """
    x: tensor-4d
    dtype: complex
    axis = (1, 3)
    """
    obj.dtype = None
    obj.types = [np.complex128]
    x_data = (np.random.rand(4, 5, 4, 4) * 10 - 5) * 1j + np.random.rand(4, 5, 4, 4) * 10 - 5
    res = np.fft.fftshift(x_data, axes=(1, 3))
    obj.run(res=res, x=x_data, axes=(1, 3))
