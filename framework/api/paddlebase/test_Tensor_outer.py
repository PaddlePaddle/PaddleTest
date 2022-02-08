#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_outer
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestTensorOuter(APIBase):
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
        # self.enable_backward = False
        # self.delta = 1e-2


obj = TestTensorOuter(paddle.Tensor.outer)


@pytest.mark.api_base_outer_vartype
def test_outer_base():
    """
    base
    """
    x = randtool("float", -4, 4, (1, 4))
    y = randtool("float", -2, 2, (1, 4))
    res = np.outer(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_outer_parameters
def test_outer0():
    """
    default
    """
    x = randtool("float", -4, 4, (4, 1))
    y = randtool("float", -2, 2, (4, 1))
    res = np.outer(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_outer_parameters
def test_outer1():
    """
    x, y: 1d-tensor
    """
    x = randtool("float", -4, 4, (4,))
    y = randtool("float", -2, 2, (4,))
    res = np.outer(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_outer_parameters
def test_outer2():
    """
    x, y: 3d-tensor
    """
    x = randtool("float", -4, 4, (4, 2, 3))
    y = randtool("float", -2, 2, (4, 2, 3))
    res = np.outer(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_outer_parameters
def test_outer3():
    """
    x, y: 2d-tensor, dim different
    """
    x = randtool("float", -4, 4, (4, 2))
    y = randtool("float", -2, 2, (2, 3, 4))
    res = np.outer(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_outer_parameters
def test_outer4():
    """
    x, y: 4d-tensor
    """
    x = randtool("float", -4, 4, (4, 2, 5, 2))
    y = randtool("float", -2, 2, (2, 3, 4, 4))
    res = np.outer(x, y)
    obj.run(res=res, x=x, y=y)
