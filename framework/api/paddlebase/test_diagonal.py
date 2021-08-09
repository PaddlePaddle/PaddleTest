#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diagonal
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestDiagonal(APIBase):
    """
    test Diagonal
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestDiagonal(paddle.diagonal)
obj1 = TestDiagonal(paddle.diagonal)
obj1.types = [np.int32, np.int64]


@pytest.mark.api_base_diagonal_vartype
def test_diagonal_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6])
    res = np.diagonal(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_diagonal_vartype
def test_diagonal_base1():
    """
    base int
    """
    x = randtool("int", -5, 5, [6, 6])
    res = np.diagonal(x)
    obj1.base(res=res, x=x)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal():
    """
    offset = 1
    """
    x = randtool("int", -5, 5, [6, 6])
    offset = 1
    res = np.diagonal(x, offset=offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal1():
    """
    offset = -1
    """
    x = randtool("int", -5, 5, [6, 6])
    offset = -1
    res = np.diagonal(x, offset=offset)
    obj.run(res=res, x=x, offset=offset)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal2():
    """
    3d tensor
    """
    x = randtool("int", -5, 5, [6, 6, 6])
    res = np.diagonal(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal3():
    """
    4d tensor
    """
    x = randtool("int", -5, 5, [6, 6, 6, 6])
    res = np.diagonal(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal4():
    """
    5d tensor
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    res = np.diagonal(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal5():
    """
    axis1 = 0, axis2 = 3
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    axis1 = 0
    axis2 = 3
    res = np.diagonal(x, axis1=axis1, axis2=axis2)
    obj.run(res=res, x=x, axis1=axis1, axis2=axis2)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal6():
    """
    axis1 = 2, axis2 = 3
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    axis1 = 2
    axis2 = 3
    res = np.diagonal(x, axis1=axis1, axis2=axis2)
    obj.run(res=res, x=x, axis1=axis1, axis2=axis2)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal7():
    """
    axis1 = 3, axis2 = 4
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    axis1 = 3
    axis2 = 4
    res = np.diagonal(x, axis1=axis1, axis2=axis2)
    obj.run(res=res, x=x, axis1=axis1, axis2=axis2)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal8():
    """
    axis1 = 4, axis2 = 2
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    axis1 = 4
    axis2 = 2
    res = np.diagonal(x, axis1=axis1, axis2=axis2)
    obj.run(res=res, x=x, axis1=axis1, axis2=axis2)


@pytest.mark.api_base_diagonal_parameters
def test_diagonal9():
    """
    axis1 = -1, axis2 = 2
    """
    x = randtool("int", -5, 5, [6, 6, 6, 2, 2])
    axis1 = -1
    axis2 = 2
    res = np.diagonal(x, axis1=axis1, axis2=axis2)
    obj.run(res=res, x=x, axis1=axis1, axis2=axis2)
