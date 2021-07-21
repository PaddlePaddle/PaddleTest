#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sinh
"""
from apibase import APIBase
from apibase import randtool
import paddle
import numpy as np
import pytest


class TestSinh(APIBase):
    """
    test sinh
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


obj = TestSinh(paddle.sinh)


@pytest.mark.api_base_sinh_vartype
def test_sinh_base():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float64)
    res = np.sinh(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_sinh_parameters
def test_sinh_1():
    """
    x.shape=(1, 2)
    float64
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float64)
    res = np.sinh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sinh_parameters
def test_sinh_2():
    """
    x.shape=(1, 2)
    float32
    """
    x = randtool("float", -10, 10, [1, 2]).astype(np.float32)
    res = np.sinh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sinh_parameters
def test_sinh_3():
    """
    x.shape=(2, 2)
    float64
    """
    x = randtool("float", -10, 10, [2, 2]).astype(np.float64)
    res = np.sinh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sinh_parameters
def test_sinh_4():
    """
    x.shape=(1, )
    """
    x = np.array([8.931])
    res = np.sinh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sinh_parameters
def test_sinh_5():
    """
    x.shape=(2, 3, 2, 2)
    float32
    """
    x = randtool("float", -10, 10, [2, 3, 2, 2]).astype(np.float32)
    res = np.sinh(x)
    obj.run(res=res, x=x)
