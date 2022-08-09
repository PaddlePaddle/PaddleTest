#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test log1p
"""

from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import pytest


class TestLog1p(APIBase):
    """
    test log1p
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


obj = TestLog1p(paddle.log1p)


@pytest.mark.api_base_log1p_vartype
def test_log1p_base():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log1p(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_1():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float64)
    res = np.log1p(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_2():
    """
    x.shape=(1, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [1, 2]).astype(np.float32)
    res = np.log1p(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_3():
    """
    x.shape=(2, 2)
    float64
    """
    x = randtool("float", 0.5, 20, [2, 2]).astype(np.float64)
    res = np.log1p(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_4():
    """
    x.shape=(1, )
    """
    x = np.array([15.54])
    res = np.log1p(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_5():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", 0.5, 20, [2, 3, 2, 2]).astype(np.float32)
    res = np.log1p(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_log1p_parameters
def test_log1p_6():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", -0.8, 20, [2, 3, 2, 2]).astype(np.float32)
    res = np.log1p(x)
    obj.run(res=res, x=x)
