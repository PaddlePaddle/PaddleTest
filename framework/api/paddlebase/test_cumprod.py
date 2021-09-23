#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cumprod
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestDist(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-2


obj = TestDist(paddle.cumprod)


@pytest.mark.api_base_cumprod_vartype
def test_cumprod_base():
    """
    base: dim = 0
    """
    x = np.arange(12).reshape((3, 4))

    res = np.array([[0, 1, 2, 3], [0, 5, 12, 21], [0, 45, 120, 231]])
    obj.base(res=res, x=x, dim=0)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod0():
    """
    default:
    dim = 1
    """
    x = np.arange(12, 24).reshape((3, 4))

    res = np.array([[12.0, 156.0, 2184.0, 32760.0], [16.0, 272.0, 4896.0, 93024.0], [20.0, 420.0, 9240.0, 212520.0]])
    obj.run(res=res, x=x, dim=1)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod1():
    """
    default:
    dim = -1
    """
    x = np.arange(12, 24).reshape((3, 4))

    res = np.array([[12.0, 156.0, 2184.0, 32760.0], [16.0, 272.0, 4896.0, 93024.0], [20.0, 420.0, 9240.0, 212520.0]])
    obj.run(res=res, x=x, dim=-1)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod2():
    """
    Multidimensional (2, 3, 2)
    dim = 0
    """
    x = np.arange(12).reshape((2, 3, 2))

    res = np.array([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], [[0.0, 7.0], [16.0, 27.0], [40.0, 55.0]]])
    obj.run(res=res, x=x, dim=0)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod3():
    """
    Multidimensional (2, 3, 2)
    dim = 1
    """
    x = np.arange(12).reshape((2, 3, 2))

    res = np.array([[[0.0, 1.0], [0.0, 3.0], [0.0, 15.0]], [[6.0, 7.0], [48.0, 63.0], [480.0, 693.0]]])
    obj.run(res=res, x=x, dim=1)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod3():
    """
    Multidimensional (2, 3, 2)
    dim = -2
    """
    x = np.arange(12).reshape((2, 3, 2))

    res = np.array([[[0.0, 1.0], [0.0, 3.0], [0.0, 15.0]], [[6.0, 7.0], [48.0, 63.0], [480.0, 693.0]]])
    obj.run(res=res, x=x, dim=-2)


@pytest.mark.api_base_cumprod_parameters
def test_cumprod4():
    """
    vector
    """
    x = np.arange(12)
    res = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obj.run(res=res, x=x, dim=0)
