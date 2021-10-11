#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_pinv
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestPinv(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False


obj = TestPinv(paddle.linalg.pinv)


@pytest.mark.api_linalg_pinv_vartype
def test_pinv_base0():
    """
    base: svd
    """
    # x = np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    x = randtool("float", -2, 4, (3, 4))
    res = np.linalg.pinv(x)
    obj.types = [np.float32, np.float64]
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_pinv_vartype
def test_pinv_base1():
    """
    base: eig
    x.value: complex
    """
    x = np.array([[2 - 1j, 4 + 5j], [4 - 5j, 7]])
    res = np.linalg.pinv(x, hermitian=True)
    obj.types = [np.complex64, np.complex128]
    obj.base(res=res, x=x, hermitian=True)


@pytest.mark.api_linalg_pinv_vartype
def test_cond_base1():
    """
    base: eig
    """
    # x = np.array([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    x = randtool("float", -2, 4, (2, 4))
    x = np.dot(x, x.T)
    res = np.linalg.pinv(x)
    obj.types = [np.float32, np.float64, np.complex64]
    obj.base(res=res, x=x, hermitian=True)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv0():
    """
    default
    """
    x = randtool("float", -2, 4, [2, 40])
    res = np.linalg.pinv(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv1():
    """
    x: multiple dimension
    """
    x = randtool("float", 2, 21, [2, 4, 40])
    res = np.linalg.pinv(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv2():
    """
    x: multiple dimension
    rcondc = 0.5
    """
    x = randtool("float", 2, 21, [2, 4, 40])
    res = np.linalg.pinv(x, rcond=0.5)
    obj.run(res=res, x=x, rcond=0.5)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv3():
    """
    eig
    """
    x = randtool("float", -2, 4, (2, 41))
    x = np.dot(x, x.T)
    res = np.linalg.pinv(x)
    obj.run(res=res, x=x, hermitian=True)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv4():
    """
    eig
    x: multiple dimension
    """
    x = randtool("float", -2, 4, (4, 2, 41))
    x = np.matmul(x, np.transpose(x, (0, 2, 1)))
    res = np.linalg.pinv(x)
    obj.run(res=res, x=x, hermitian=True)


@pytest.mark.api_linalg_pinv_parameters
def test_pinv5():
    """
    eig
    x: multiple dimension
    rcond = 5
    """
    x = randtool("float", -2, 4, (4, 2, 41))
    x = np.matmul(x, np.transpose(x, (0, 2, 1)))
    res = np.linalg.pinv(x, rcond=5)
    obj.run(res=res, x=x, rcond=5, hermitian=True)
