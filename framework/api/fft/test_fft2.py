#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fft2
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFFt2(APIBase):
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


obj = TestFFt2(paddle.fft.fft2)


@pytest.mark.api_fft_fft2_vartype
def test_fft2_base():
    """
    base
    dtype: float
    """
    np.random.seed(33)
    x = np.random.rand(4, 4)
    res = np.array(
        [
            [
                (7.501306584423414 + 0j),
                (0.31964998090497876 + 1.1171063833691184j),
                (-0.6249497790229205 + 0j),
                (0.31964998090497876 - 1.1171063833691184j),
            ],
            [
                (-1.1557384478971886 - 0.4505820931029272j),
                (0.25859804729780966 - 1.0312981969701076j),
                (-0.8163149404282567 - 0.8431466666670808j),
                (-0.01430736814560174 - 0.8383255115001054j),
            ],
            [
                (0.28907448085882637 + 0j),
                (-1.2136633622924302 - 1.6828366185546986j),
                (2.0542829338657356 + 0j),
                (-1.2136633622924302 + 1.6828366185546986j),
            ],
            [
                (-1.1557384478971886 + 0.4505820931029272j),
                (-0.01430736814560174 + 0.8383255115001054j),
                (-0.8163149404282567 + 0.8431466666670808j),
                (0.25859804729780966 + 1.0312981969701076j),
            ],
        ]
    )
    obj.base(res=res, x=x)


@pytest.mark.api_fft_fft2_parameters
def test_fft2_0():
    """
    default
    """
    x_data = np.mgrid[:2, :4][1]
    res = np.array(
        [[12.0 + 0.0j, -4.0 + 4.0j, -4.0 + 0.0j, -4.0 - 4.0j], [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fft2_parameters
def test_fft2_1():
    """
    x: complex
    """
    obj.dtype = None
    obj.types = [np.complex128, np.complex64]
    x_data = np.exp(3j * np.pi * np.arange(12).reshape(3, 4) / 7)
    res = np.array(
        [
            [
                (0.678447933946106 + 1.408811651299378j),
                (2.8758967800842363 - 8.218840801895883j),
                (1.1234898018587356 - 0.5410441730642647j),
                (0.9260409557205965 + 0.3240357450613556j),
            ],
            [
                (-0.38325498822205617 + 0.5621318239715889j),
                (3.722576607412038 - 0.7043494956566969j),
                (0.4482851706612676 + 0.3056356542635312j),
                (0.07936112839279974 + 0.41943365036049524j),
            ],
            [
                (-1.2010061531385652 - 0.09000296877691061j),
                (4.3747114001605425 + 5.083506872754761j),
                (-0.07177497252000231 + 0.9577704470120258j),
                (-0.5727736643556987 + 0.4929115946706193j),
            ],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fft2_parameters
def test_fft2_2():
    """
    input dim = 3
    x = np.mgrid[:2, :4]
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.mgrid[:2, :4]
    res = np.array(
        [
            [[4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [-4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
            [[12.0 + 0.0j, -4.0 + 4.0j, -4.0 + 0.0j, -4.0 - 4.0j], [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
        ]
    )

    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fft2_parameters
def test_fft2_3():
    """
    input dim = 4
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.mgrid[:2, :4, -4:-2:2j]
    res = np.array(
        [
            [[[0j, 0j], [0j, 0j], [0j, 0j], [0j, 0j]], [[(8 + 0j), 0j], [0j, 0j], [0j, 0j], [0j, 0j]]],
            [
                [[(12 + 0j), 0j], [(-4 + 4j), 0j], [(-4 + 0j), 0j], [(-4 - 4j), 0j]],
                [[(12 + 0j), 0j], [(-4 + 4j), 0j], [(-4 + 0j), 0j], [(-4 - 4j), 0j]],
            ],
            [
                [[(-24 + 0j), (-8 + 0j)], [0j, 0j], [0j, 0j], [0j, 0j]],
                [[(-24 + 0j), (-8 + 0j)], [0j, 0j], [0j, 0j], [0j, 0j]],
            ],
        ]
    )
    obj.run(res=res, x=x_data)


@pytest.mark.api_fft_fft2_parameters
def test_fft2_4():
    """
    input dim = 4
    s = (1, 2)
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.mgrid[4:10:3j, 2:4:2j, -4:-2:2j]
    res = np.array(
        [
            [[[(8 + 0j), 0j]], [[(14 + 0j), 0j]], [[(20 + 0j), 0j]]],
            [[[(4 + 0j), 0j]], [[(4 + 0j), 0j]], [[(4 + 0j), 0j]]],
            [[[(-6 + 0j), (-2 + 0j)]], [[(-6 + 0j), (-2 + 0j)]], [[(-6 + 0j), (-2 + 0j)]]],
        ]
    )
    obj.run(res=res, x=x_data, s=(1, 2))


@pytest.mark.api_fft_fft2_parameters
def test_fft2_5():
    """
    x: tensor-3D
    dim = (1, 2)
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.exp(3j * np.pi * np.arange(16).reshape((2, 4, 2)))
    res = np.array(
        [
            [
                [0.0000e00 + 1.4696e-15j, 8.0000e00 - 1.0287e-14j],
                [3.5527e-15 - 3.5527e-15j, -6.1356e-16 + 6.4919e-15j],
                [0.0000e00 + 7.1054e-15j, 0.0000e00 - 4.1663e-15j],
                [-3.5527e-15 - 3.5527e-15j, 6.1356e-16 + 6.4919e-15j],
            ],
            [
                [0.0000e00 + 1.4696e-15j, 8.0000e00 - 3.3800e-14j],
                [-1.4211e-14 + 7.8886e-31j, -1.1272e-14 + 2.9392e-15j],
                [0.0000e00 + 1.4211e-14j, 0.0000e00 - 1.1272e-14j],
                [1.4211e-14 + 7.8886e-31j, 1.1272e-14 + 2.9392e-15j],
            ],
        ]
    )
    obj.run(res=res, x=x_data, axes=(1, 2))


@pytest.mark.api_fft_fft2_parameters
def test_fft2_6():
    """
    x: tensor-3D
    dim = (1, 2)
    norm = 'forward'
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.exp(3j * np.pi * np.arange(16).reshape((2, 4, 2)))
    res = np.array(
        [
            [
                [0.0000e00 + 1.8370e-16j, 1.0000e00 - 1.2859e-15j],
                [4.4409e-16 - 4.4409e-16j, -7.6695e-17 + 8.1148e-16j],
                [0.0000e00 + 8.8818e-16j, 0.0000e00 - 5.2078e-16j],
                [-4.4409e-16 - 4.4409e-16j, 7.6695e-17 + 8.1148e-16j],
            ],
            [
                [0.0000e00 + 1.8370e-16j, 1.0000e00 - 4.2250e-15j],
                [-1.7764e-15 + 9.8608e-32j, -1.4090e-15 + 3.6739e-16j],
                [0.0000e00 + 1.7764e-15j, 0.0000e00 - 1.4090e-15j],
                [1.7764e-15 + 9.8608e-32j, 1.4090e-15 + 3.6739e-16j],
            ],
        ]
    )
    obj.run(res=res, x=x_data, axes=(1, 2), norm="forward")


@pytest.mark.api_fft_fft2_parameters
def test_fft2_7():
    """
    x: tensor-3D
    dim = (1, 2)
    norm = 'ortho'
    """
    obj.types = [np.complex128, np.complex64]
    x_data = np.exp(3j * np.pi * np.arange(16).reshape((2, 4, 2)))
    res = np.array(
        [
            [
                [5.195736337412959e-16j, (2.82842712474619 - 3.6370154361890715e-15j)],
                [(1.25607396694702e-15 - 1.2560739669470197e-15j), (-2.169266994644281e-16 + 2.2952212344296118e-15j)],
                [2.51214793389404e-15j, -1.473000666411448e-15j],
                [(-1.25607396694702e-15 - 1.2560739669470197e-15j), (2.169266994644281e-16 + 2.2952212344296118e-15j)],
            ],
            [
                [5.195736337412959e-16j, (2.82842712474619 - 1.1950193576049806e-14j)],
                [
                    (-5.024295867788079e-15 + 2.9582283945787943e-31j),
                    (-3.985148600305487e-15 + 1.0391472674825918e-15j),
                ],
                [5.024295867788081e-15j, -3.985148600305488e-15j],
                [(5.024295867788079e-15 + 2.9582283945787943e-31j), (3.985148600305487e-15 + 1.0391472674825918e-15j)],
            ],
        ]
    )
    obj.run(res=res, x=x_data, axes=(1, 2), norm="ortho")
