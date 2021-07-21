#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test log10
"""

from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import pytest


class TestLog10(APIBase):
    """
    test log10
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestLog10(paddle.log10)


@pytest.mark.api_base_log10_vartype
def test_log10_base():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log10(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_log10_parameters
def test_log10_1():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log10(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log10_parameters
def test_log10_2():
    """
    x.shape=(1, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float32)
    res = np.log10(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log10_parameters
def test_log10_3():
    """
    x.shape=(2, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [2, 2]).astype(np.float64)
    res = np.log10(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log10_parameters
def test_log10_4():
    """
    x.shape=(1, )
    """
    x = np.array([15.54])
    res = np.log10(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log10_parameters
def test_log10_5():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [2, 3, 2, 2]).astype(np.float32)
    res = np.log10(x)
    obj.run(res=res, x=x)
