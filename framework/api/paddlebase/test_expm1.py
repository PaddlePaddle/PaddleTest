#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test expm1
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestExpm1(APIBase):
    """
    test expm1
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


obj = TestExpm1(paddle.expm1)
# test float16
obj1 = TestExpm1(paddle.expm1)
obj1.types = [np.float16]
obj1.delta = 1e-3
obj1.enable_backward = False


@pytest.mark.api_base_expm1_vartype
def test_expm1_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6]).astype(np.float32)
    res = np.expm1(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_expm1_parameters
def test_expm1_0():
    """
    base float 3-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6])
    res = np.expm1(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_expm1_parameters
def test_expm1_1():
    """
    base float 4-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6, 6])
    res = np.expm1(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_expm1_parameters
def test_expm1_2():
    """
    base float 5-D tensor
    """
    x = randtool("float", -5, 5, [3, 6, 6, 6, 6])
    res = np.expm1(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_expm1_parameters
def test_expm1_3():
    """
    float16
    """
    x = randtool("float", -1, 1, [3, 3, 3]).astype(np.float16)
    res = np.expm1(x)
    obj1.run(res=res, x=x)
