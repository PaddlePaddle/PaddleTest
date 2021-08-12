#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test trunc
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestTrunc(APIBase):
    """
    test trunc
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


obj = TestTrunc(paddle.trunc)
# test float16
obj1 = TestTrunc(paddle.trunc)
obj1.types = [np.int32, np.int64]


@pytest.mark.api_base_trunc_vartype
def test_trunc_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6])
    res = np.trunc(x)
    obj.base(res=res, input=x)


@pytest.mark.api_base_trunc_vartype
def test_trunc_base():
    """
    base int
    """
    x = randtool("int", -5, 5, [6, 6])
    res = np.trunc(x)
    obj1.base(res=res, input=x)


@pytest.mark.api_base_trunc_parameters
def test_trunc():
    """
    base float 3-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6])
    res = np.trunc(x)
    obj.run(res=res, input=x)


@pytest.mark.api_base_trunc_parameters
def test_trunc1():
    """
    base float 4-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6, 6])
    res = np.trunc(x)
    obj.run(res=res, input=x)


@pytest.mark.api_base_trunc_parameters
def test_trunc2():
    """
    base float 5-D tensor
    """
    x = randtool("float", -5, 5, [3, 6, 6, 6, 6])
    res = np.trunc(x)
    obj.run(res=res, input=x)
