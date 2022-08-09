#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_moveaxis
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestMoveAxis(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


obj = TestMoveAxis(paddle.moveaxis)


@pytest.mark.api_base_moveaxis_vartype
def test_moveaxis_base0():
    """
    base
    type: float
    """
    x = randtool("float", -2, 2, (4, 2))
    res = np.transpose(x, (1, 0))
    obj.base(res=res, x=x, source=0, destination=1)


@pytest.mark.api_base_moveaxis_vartype
def test_moveaxis_base1():
    """
    base
    type: complex
    """
    obj.types = [np.complex64, np.complex128]
    obj.static = False
    x = randtool("float", -2, 2, (4, 2)) + randtool("float", 0, 1, (4, 2)) * 1j
    res = np.transpose(x, (1, 0))
    obj.base(res=res, x=x, source=0, destination=1)


@pytest.mark.api_base_moveaxis_vartype
def test_moveaxis_base2():
    """
    base
    type: int
    """
    obj.types = [np.int32, np.int64]
    obj.static = True
    x = randtool("int", -2, 2, (4, 2))
    res = np.transpose(x, (1, 0))
    obj.base(res=res, x=x, source=0, destination=1)


@pytest.mark.api_base_moveaxis_parameters
def test_moveaxis0():
    """
    x: 3d-tensor
    """
    obj.types = [np.float32, np.float64]
    x = randtool("float", -2, 2, (4, 2, 3))
    res = np.transpose(x, (1, 2, 0))
    obj.run(res=res, x=x, source=0, destination=2)


@pytest.mark.api_base_moveaxis_parameters
def test_moveaxis1():
    """
    x: 4d-tensor
    """
    obj.types = [np.float32, np.float64]
    x = randtool("float", -2, 2, (4, 2, 3, 5))
    res = np.transpose(x, (1, 2, 0, 3))
    obj.run(res=res, x=x, source=0, destination=2)


@pytest.mark.api_base_moveaxis_parameters
def test_moveaxis2():
    """
    x: 5d-tensor
    """
    obj.types = [np.float32, np.float64]
    x = randtool("float", -2, 2, (4, 2, 3, 5, 7))
    res = np.transpose(x, (1, 2, 0, 3, 4))
    obj.run(res=res, x=x, source=0, destination=2)


@pytest.mark.api_base_moveaxis_parameters
def test_moveaxis3():
    """
    x: 5d-tensor
    source, destination: tuple
    """
    obj.types = [np.float32, np.float64]
    x = randtool("float", -2, 2, (4, 2, 3, 5, 7))
    res = np.transpose(x, (2, 3, 0, 1, 4))
    obj.run(res=res, x=x, source=(0, 1), destination=(2, 3))
