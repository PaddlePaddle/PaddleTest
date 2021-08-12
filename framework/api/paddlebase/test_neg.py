#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test neg
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestNeg(APIBase):
    """
    test Neg
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


obj = TestNeg(paddle.neg)
obj1 = TestNeg(paddle.neg)
obj1.types = [np.int8, np.int16, np.int32, np.int64]


@pytest.mark.api_base_neg_vartype
def test_neg_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6])
    res = np.negative(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_neg_vartype
def test_neg_base1():
    """
    base int
    """
    x = randtool("int", -5, 5, [6, 6])
    res = np.negative(x)
    obj1.base(res=res, x=x)


@pytest.mark.api_base_neg_parameters
def test_neg():
    """
    test zero
    """
    x = np.zeros(shape=[6, 6, 6])
    res = np.negative(x)
    obj.run(res=res, x=x)
