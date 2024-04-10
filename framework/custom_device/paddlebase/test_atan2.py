#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test atan2
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestAtan2(APIBase):
    """
    test atan2
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.delta = 1e-6
        # self.rtol = 1e-6
        self.enable_backward = True


obj = TestAtan2(paddle.atan2)
# test float16
obj1 = TestAtan2(paddle.atan2)
obj1.types = [np.float16]
obj1.delta = 1e-3
obj1.rtol = 1e-3
obj1.enable_backward = False
# test int type
obj2 = TestAtan2(paddle.atan2)
obj2.types = [np.int32, np.int64]


@pytest.mark.api_base_atan2_vartype
def test_atan2_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6])
    y = randtool("float", -5, 5, [6, 6])
    res = np.arctan2(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_atan2_vartype
def test_atan2_base1():
    """
    base float16
    """
    x = randtool("float", -5, 5, [6, 6]).astype(np.float16)
    y = randtool("float", -5, 5, [6, 6]).astype(np.float16)
    res = np.arctan2(x, y)
    obj1.base(res=res, x=x, y=y)


@pytest.mark.api_base_atan2_vartype
def test_atan2_base2():
    """
    base int
    """
    x = randtool("int", -5, 5, [6, 6])
    y = randtool("int", -5, 5, [6, 6])
    res = np.arctan2(x, y)
    obj2.base(res=res, x=x, y=y)


@pytest.mark.api_base_atan2_parameters
def test_atan2_0():
    """
    base float 3-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6])
    y = randtool("float", -5, 5, [6, 6, 6])
    res = np.arctan2(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_atan2_parameters
def test_atan2_1():
    """
    base float 4-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6, 6])
    y = randtool("float", -5, 5, [6, 6, 6, 6])
    res = np.arctan2(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_atan2_parameters
def test_atan2_2():
    """
    base float 5-D tensor
    """
    x = randtool("float", -5, 5, [3, 6, 6, 6, 6])
    y = randtool("float", -5, 5, [3, 6, 6, 6, 6])
    res = np.arctan2(x, y)
    obj.run(res=res, x=x, y=y)
