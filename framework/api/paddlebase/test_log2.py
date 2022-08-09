#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test log2
"""

from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import pytest


class TestLog2(APIBase):
    """
    test log2
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


obj = TestLog2(paddle.log2)


@pytest.mark.api_base_log2_vartype
def test_log2_base():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log2(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_log2_parameters
def test_log2_1():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log2(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log2_parameters
def test_log2_2():
    """
    x.shape=(1, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float32)
    res = np.log2(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log2_parameters
def test_log2_3():
    """
    x.shape=(2, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [2, 2]).astype(np.float64)
    res = np.log2(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log2_parameters
def test_log2_4():
    """
    x.shape=(1, )
    """
    x = np.array([15.54])
    res = np.log2(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log2_parameters
def test_log2_5():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [2, 3, 2, 2]).astype(np.float32)
    res = np.log2(x)
    obj.run(res=res, x=x)
