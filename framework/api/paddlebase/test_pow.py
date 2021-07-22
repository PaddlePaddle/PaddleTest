#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test pow
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestPow(APIBase):
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
        # self.enable_backward = True


obj = TestPow(paddle.pow)


@pytest.mark.api_base_pow_vartype
def test_pow_base():
    """
    base
    """
    x = randtool("int", 1, 2, [2, 2, 2])
    y = randtool("int", 1, 2, [2, 2, 2])
    res = np.power(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_pow_parameters
def test_pow():
    """
    default
    """
    x = randtool("float", 1, 2, [2, 2, 2])
    y = randtool("float", 1, 2, [2, 2, 2])
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_pow_parameters
def test_pow1():
    """
    y = 0
    """
    x = randtool("float", 1, 2, [2, 2, 2])
    y = 0
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_pow_parameters
def test_pow2():
    """
    x < 0
    """
    x = randtool("float", -4, -2, [2, 2, 2])
    y = randtool("float", 1, 2, [2, 2, 2])
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_pow_parameters
def test_pow3():
    """
    broadcast
    """
    x = randtool("float", -4, -2, [2, 2, 2])
    y = randtool("float", 1, 2, [2])
    res = np.power(x, y)
    obj.run(res=res, x=x, y=y)
