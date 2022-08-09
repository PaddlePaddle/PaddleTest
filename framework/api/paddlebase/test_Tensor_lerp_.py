#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_lerp_
"""

import sys
import os
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2


class TestLerp(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestLerp(paddle.Tensor.lerp_)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_vartype
def test_lerp_base():
    """
    base
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([5.0, 6.0, 7.0, 8.0])
    w = 0.5
    res = x + w * (y - x)
    obj.base(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_0():
    """
    default
    """
    x = np.random.rand(10)
    y = np.random.rand(10) * 4
    w = 0.5
    res = x + w * (y - x)
    obj.base(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_1():
    """
    x: 2-d tensor
    """
    x = np.random.rand(10, 3)
    y = np.random.rand(10, 3) * 4
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_2():
    """
    x: 3-d tensor
    """
    x = np.random.rand(10, 3, 4)
    y = np.random.rand(10, 3, 4) * 4
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_3():
    """
    x: 4-d tensor
    """
    x = np.random.rand(4, 4, 3, 4)
    y = np.random.rand(4, 4, 3, 4) * 4
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_4():
    """
    x: 4-d tensor
    w = 4.
    """
    x = np.random.rand(4, 4, 3, 4)
    y = np.random.rand(4, 4, 3, 4) * 4
    w = 4.0
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp_5():
    """
    x: 4-d tensor
    w = -4.
    """
    x = np.random.rand(4, 4, 3, 4)
    y = np.random.rand(4, 4, 3, 4) * 4
    w = -4.0
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)
