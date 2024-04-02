#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cross
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCross(APIBase):
    """
    test cross
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestCross(paddle.cross)


@pytest.mark.api_base_cross_vartype
def test_cross_base():
    """
    test base
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    res = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_cross_parameters
def test_cross():
    """
    default
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    res = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_cross_parameters
def test_cross1():
    """
    dim=1
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    res = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    obj.run(res=res, x=x, y=y, axis=1)


@pytest.mark.api_base_cross_parameters
def test_cross2():
    """
    dim=-1
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    res = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    obj.run(res=res, x=x, y=y, axis=-1)


@pytest.mark.api_base_cross_parameters
def test_cross3():
    """
    dim=0
    """
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    res = np.cross(x, y, axis=0)
    obj.run(res=res, x=x, y=y, axis=0)


@pytest.mark.api_base_cross_parameters
def test_cross4():
    """
    dim=0 random x,y
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.cross(x, y, axis=0)
    obj.run(res=res, x=x, y=y, axis=0)


@pytest.mark.api_base_cross_parameters
def test_cross5():
    """
    dim=1 random x,y
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.cross(x, y, axis=1)
    obj.run(res=res, x=x, y=y, axis=1)


@pytest.mark.api_base_cross_parameters
def test_cross6():
    """
    dim=2 random x,y
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.cross(x, y, axis=2)
    obj.run(res=res, x=x, y=y, axis=2)
