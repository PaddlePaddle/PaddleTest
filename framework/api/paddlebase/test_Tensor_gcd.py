#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_gcd
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestTensorGcd(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-2


obj = TestTensorGcd(paddle.Tensor.gcd)


@pytest.mark.api_base_gcd_vartype
def test_gcd_base():
    """
    base
    """
    x = randtool("int", -4, 40, (1,))
    y = randtool("int", -2, 2, (1,))
    res = np.gcd(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_gcd_parameters
def test_gcd0():
    """
    default
    """
    x = randtool("int", -4, 14, (1,))
    y = randtool("int", -21, 2, (1,))
    res = np.gcd(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_gcd_parameters
def test_gcd1():
    """
    x, y: 2-d tensor
    """
    x = randtool("int", -4, 14, (4, 5))
    y = randtool("int", -21, 2, (4, 5))
    res = np.gcd(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_gcd_parameters
def test_gcd2():
    """
    x, y: 3-d tensor
    """
    x = randtool("int", -4, 14, (2, 4, 5))
    y = randtool("int", -21, 2, (2, 4, 5))
    res = np.gcd(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_gcd_parameters
def test_gcd3():
    """
    x, y: 4-d tensor
    """
    x = randtool("int", -4, 14, (6, 2, 4, 5))
    y = randtool("int", -21, 2, (6, 2, 4, 5))
    res = np.gcd(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_gcd_parameters
def test_gcd4():
    """
    x, y: broadcast
    """
    x = randtool("int", -4, 14, (6, 1, 4, 5))
    y = randtool("int", -21, 2, (2, 1, 5))
    res = np.gcd(x, y)
    obj.run(res=res, x=x, y=y)
