#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fill_
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFill_(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.types = [np.float32]
        # self.debug = True
        self.static = False
        # enable check grad
        self.enable_backward = False


obj = TestFill_(paddle.Tensor.fill_)


@pytest.mark.api_base_fill_vartype
def test_fill_base():
    """
    base
    """
    x = randtool("int", -5, 4, (4, 4))
    res = np.array([4] * 16).reshape(4, 4)
    obj.base(res=res, x=x, value=4)


@pytest.mark.api_base_fill_parameters
def test_fill0():
    """
    x: vector
    """
    x = randtool("float", -5, 4, (16,))
    res = np.array([4.2] * 16).reshape(16)
    obj.run(res=res, x=x, value=4.2)


@pytest.mark.api_base_fill_parameters
def test_fill1():
    """
    x: matrix
    """
    x = randtool("float", -5, 4, (16, 16))
    res = np.array([41.2] * 256).reshape(16, 16)
    obj.run(res=res, x=x, value=41.2)


@pytest.mark.api_base_fill_parameters
def test_fill2():
    """
    x: multiple dimension tensor
    """
    x = randtool("float", -5, 4, (10, 16, 16))
    res = np.array([41.2] * 2560).reshape((10, 16, 16))
    obj.run(res=res, x=x, value=41.2)
