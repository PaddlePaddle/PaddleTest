#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_hfft
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFFthfft(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.complex128, np.complex64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestFFthfft(paddle.fft.hfft)


@pytest.mark.api_fft_hfft_vartype
def test_hfft_base():
    """
    base
    """
    x = np.exp(3j * np.pi * np.arange(10))
    res = np.array(
        [
            -1.1102e-16,
            1.3323e-15,
            6.6571e-15,
            -1.4211e-14,
            1.0376e-14,
            3.6309e-15,
            -6.4826e-15,
            -4.8621e-15,
            2.1093e-14,
            1.8000e01,
            -2.1177e-14,
            5.4443e-15,
            6.5936e-15,
            -3.1315e-15,
            -1.0443e-14,
            1.4211e-14,
            -6.5053e-15,
            -2.4139e-15,
        ]
    )
    obj.base(res=res, x=x)


@pytest.mark.api_fft_hfft_parameters
def test_hfft0():
    """
    default
    x: tensor-2d
    """
    x_data = np.mgrid[:2:2j, :4:4j][1]
    res = np.array(
        [
            [12.0, -5.33333333, 0.0, -1.33333333, 0.0, -5.33333333],
            [12.0, -5.33333333, 0.0, -1.33333333, 0.0, -5.33333333],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_hfft_parameters
def test_hfft1():
    """
    default
    x: tensor-3d
    """
    x_data = np.mgrid[:2:2j, :4:4j, -4:2:3j][1]
    res = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [5.33333333, 0.0, 0.0, 0.0], [10.66666667, 0.0, 0.0, 0.0], [16.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [5.33333333, 0.0, 0.0, 0.0], [10.66666667, 0.0, 0.0, 0.0], [16.0, 0.0, 0.0, 0.0]],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_hfft_parameters
def test_hfft2():
    """
    default
    x: tensor-4d
    """
    x_data = np.mgrid[1:4:3j, 5:9:2j, 10:12:2j, 20:23:3j][1]
    res = np.array(
        [
            [[[20.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]], [[36.0, 0.0, 0.0, 0.0], [36.0, 0.0, 0.0, 0.0]]],
            [[[20.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]], [[36.0, 0.0, 0.0, 0.0], [36.0, 0.0, 0.0, 0.0]]],
            [[[20.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]], [[36.0, 0.0, 0.0, 0.0], [36.0, 0.0, 0.0, 0.0]]],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_hfft_parameters
def test_hfft3():
    """
    default
    x: tensor-4d
    n=2
    """
    x_data = np.mgrid[1:4:3j, 5:9:2j, 10:12:2j, 20:23:3j][1]
    res = np.array(
        [
            [[[10.0, 0.0], [10.0, 0.0]], [[18.0, 0.0], [18.0, 0.0]]],
            [[[10.0, 0.0], [10.0, 0.0]], [[18.0, 0.0], [18.0, 0.0]]],
            [[[10.0, 0.0], [10.0, 0.0]], [[18.0, 0.0], [18.0, 0.0]]],
        ]
    )
    obj.run(res=res, x=x_data, n=2)


@pytest.mark.api_fft_hfft_parameters
def test_hfft4():
    """
    default
    x: tensor-4d
    n=2
    axis=1
    """
    x_data = np.mgrid[1:4:3j, 5:9:2j, 10:12:2j, 20:23:3j][1]
    res = np.array(
        [
            [[[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]], [[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]],
            [[[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]], [[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]],
            [[[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]], [[-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]],
        ]
    )
    obj.run(res=res, x=x_data, n=2, axis=1)


@pytest.mark.api_fft_hfft_parameters
def test_hfft5():
    """
    default
    x: tensor-2d
    norm = forward
    """
    x_data = np.mgrid[:2:2j, :4:4j][1]
    res = np.array(
        [[2.0, -0.88888896, 0.0, -0.22222224, 0.0, -0.88888884], [2.0, -0.88888896, 0.0, -0.22222224, 0.0, -0.88888884]]
    )
    obj.run(res=res, x=x_data, norm="forward")


@pytest.mark.api_fft_hfft_parameters
def test_hfft6():
    """
    default
    x: tensor-2d
    norm = ortho
    """
    x_data = np.mgrid[:2:2j, :4:4j][1]
    res = np.array(
        [
            [4.89897919, -2.17732430, 0.0, -0.54433107, 0.0, -2.17732406],
            [4.89897919, -2.17732430, 0.0, -0.54433107, 0.0, -2.17732406],
        ]
    )
    obj.run(res=res, x=x_data, norm="ortho")
