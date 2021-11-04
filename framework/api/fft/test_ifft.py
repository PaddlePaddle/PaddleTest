#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_ifft
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestIfft(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64, np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestIfft(paddle.fft.ifft)


@pytest.mark.api_fft_ifft_vartype
def test_ifft_base():
    """
    base
    """
    x_data = np.arange(7)
    res = np.array(
        [
            (3.000000238418579 + 0j),
            (-0.5 - 1.038260817527771j),
            (-0.5 - 0.3987366855144501j),
            (-0.5 - 0.11412171274423599j),
            (-0.5 + 0.11412171274423599j),
            (-0.5 + 0.3987366855144501j),
            (-0.5 + 1.038260817527771j),
        ]
    )
    obj.base(res=res, x=x_data)


@pytest.mark.api_fft_ifft_parameters
def test_ifft0():
    """
    default
    """
    obj.dtype = np.complex128
    x_data = np.exp(3j * np.pi * np.arange(7) / 7)
    res = np.array(
        [
            (0.1428571428571429 + 0.17913719109467185j),
            (0.1428571428571427 + 0.06879637411536121j),
            (0.14285714285714282 + 1.2688263138573217e-16j),
            (0.14285714285714293 - 0.06879637411536137j),
            (0.14285714285714277 - 0.1791371910946721j),
            (0.1428571428571426 - 0.6258980382192603j),
            (0.14285714285714304 + 0.6258980382192605j),
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fft_parameters
def test_fft1():
    """
    n = 1
    """
    obj.dtype = np.complex128
    x_data = np.exp(3j * np.pi * np.arange(7) / 7)
    res = np.array([1.0 + 0.0j])
    obj.run(res=res, x=x_data, n=1)


@pytest.mark.api_fft_fft_parameters
def test_fft2():
    """
    axis = 0
    """
    obj.dtype = np.complex128
    x_data = np.exp(3j * np.pi * np.arange(7) / 7)
    res = np.array(
        [
            (0.1428571428571429 + 0.17913719109467185j),
            (0.1428571428571427 + 0.06879637411536121j),
            (0.14285714285714282 + 1.2688263138573217e-16j),
            (0.14285714285714293 - 0.06879637411536137j),
            (0.14285714285714277 - 0.1791371910946721j),
            (0.1428571428571426 - 0.6258980382192603j),
            (0.14285714285714304 + 0.6258980382192605j),
        ]
    )
    obj.run(res=res, x=x_data, axis=0)


@pytest.mark.api_fft_fft_parameters
def test_fft3():
    """
    norm = forward
    """
    obj.dtype = np.complex128
    x_data = np.exp(3j * np.pi * np.arange(7) / 7)
    res = np.array(
        [
            (1.0000000000000004 + 1.253960337662703j),
            (0.9999999999999991 + 0.4815746188075285j),
            (0.9999999999999999 + 8.881784197001252e-16j),
            (1.0000000000000007 - 0.4815746188075296j),
            (0.9999999999999993 - 1.2539603376627049j),
            (0.9999999999999983 - 4.381286267534822j),
            (1.0000000000000013 + 4.381286267534824j),
        ]
    )
    obj.run(res=res, x=x_data, norm="forward")


@pytest.mark.api_fft_fft_parameters
def test_fft4():
    """
    norm = ortho
    """
    obj.dtype = np.complex128
    x_data = np.exp(3j * np.pi * np.arange(7) / 7)
    res = np.array(
        [
            (0.37796447300922736 + 0.47395245819915616j),
            (0.37796447300922686 + 0.18201809701220698j),
            (0.37796447300922714 + 3.3569988834012605e-16j),
            (0.3779644730092275 - 0.1820180970122074j),
            (0.3779644730092269 - 0.4739524581991568j),
            (0.3779644730092266 - 1.655970555211363j),
            (0.3779644730092277 + 1.6559705552113637j),
        ]
    )
    obj.run(res=res, x=x_data, norm="ortho")


@pytest.mark.api_fft_fft_parameters
def test_fft5():
    """
    x: tensor-2D
    norm = ortho
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.exp(3j * np.pi * np.arange(6).reshape(2, 3) / 7)
    res = np.array(
        [
            [
                (0.18564817189469687 + 0.8133777861151794j),
                (0.5026792313049576 + 0.15505600787177704j),
                (1.0437234043692232 - 0.9684337939869564j),
            ],
            [
                (0.5201746184149342 - 0.6522783401511595j),
                (-0.19218770582486303 - 0.48968628824202515j),
                (-1.4079029274104502 - 0.21220722219833893j),
            ],
        ]
    )
    obj.run(res=res, x=x_data, norm="ortho")
