#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_eigvalsh
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestEigvalsh(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.complex64, np.complex128]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False


obj = TestEigvalsh(paddle.linalg.eigvalsh)


@pytest.mark.api_linalg_eigvalsh_vartype
def test_eigvalsh_base():
    """
    base
    float32 has diff
    """
    obj.delta = 1e-2
    x = randtool("float", -4, 4, (4, 4))
    x = np.dot(x, x.T)
    res = np.linalg.eigvalsh(x)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh0():
    """
    default
    """
    obj.delta = 1e-7
    x = randtool("float", -5, 5, (10, 10))
    x = np.dot(x, x.T)
    res = np.linalg.eigvalsh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh1():
    """
    x: [4, 5, 5]
    """
    obj.delta = 1e-7
    data = []
    for i in range(4):
        x = randtool("float", -5, 5, (5, 5))
        data.append(np.dot(x, x.T).tolist())
    data = np.array(data)
    res = np.linalg.eigvalsh(data)
    obj.run(res=res, x=data)


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh2():
    """
    x: [4, 5, 5]
    UPLO = 'U'
    """
    obj.delta = 1e-7
    data = []
    for i in range(4):
        x = randtool("float", -5, 5, (5, 5))
        data.append(np.dot(x, x.T).tolist())
    data = np.array(data)
    res = np.linalg.eigvalsh(data, UPLO="U")
    obj.run(res=res, x=data, UPLO="U")


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh3():
    """
    x: complex
    """
    obj.delta = 1e-7
    obj.types = [np.complex128]
    obj.dtype = np.complex128
    obj.enable_backward = False
    x0 = randtool("float", -5, 5, (10, 10))
    x1 = np.dot(x0, x0.T)
    x2 = np.triu(x0) - np.triu(x0).T
    x = x1 + x2 * 1j
    res = np.linalg.eigvalsh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh4():
    """
    x: complex, [5, 6, 6]
    """
    obj.delta = 1e-7
    obj.types = [np.complex128]
    obj.dtype = np.complex128
    obj.enable_backward = False
    data = []
    for i in range(5):
        x0 = randtool("float", -5, 5, (10, 10))
        x1 = np.dot(x0, x0.T)
        x2 = np.triu(x0) - np.triu(x0).T
        x = x1 + x2 * 1j
        data.append(x.tolist())
    data = np.array(data)
    res = np.linalg.eigvalsh(data)
    obj.run(res=res, x=data)


@pytest.mark.api_linalg_eigvalsh_parameters
def test_eigvalsh5():
    """
    x: complex, [5, 6, 6]
    UPLO = 'U'
    """
    obj.delta = 1e-7
    obj.types = [np.complex128]
    obj.dtype = np.complex128
    obj.enable_backward = False
    data = []
    for i in range(5):
        x0 = randtool("float", -5, 5, (10, 10))
        x1 = np.dot(x0, x0.T)
        x2 = np.triu(x0) - np.triu(x0).T
        x = x1 + x2 * 1j
        data.append(x.tolist())
    data = np.array(data)
    res = np.linalg.eigvalsh(data, UPLO="U")
    obj.run(res=res, x=data, UPLO="U")
