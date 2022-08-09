#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_lerp
"""

import sys
import os
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2


class TestTensorLerp(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-3


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestTensorLerp(paddle.Tensor.lerp)


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
def test_lerp0():
    """
    default
    """
    x = np.random.rand(4, 5)
    y = np.random.rand(4, 5)
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp1():
    """
    x, y: 3d-tensor
    """
    x = np.random.rand(4, 5, 4)
    y = np.random.rand(4, 5, 4)
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp2():
    """
    x, y: 4d-tensor
    """
    x = np.random.rand(4, 5, 4, 3)
    y = np.random.rand(4, 5, 4, 3)
    w = 0.5
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp3():
    """
    x, y: 4d-tensor
    weight = 1.
    """
    x = np.random.rand(4, 5, 4, 3)
    y = np.random.rand(4, 5, 4, 3)
    w = 1.0
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp4():
    """
    x, y: 4d-tensor
    weight = 0.
    """
    x = np.random.rand(4, 5, 4, 3)
    y = np.random.rand(4, 5, 4, 3)
    w = 0.0
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)


@skip_branch_is_2_2
@pytest.mark.api_base_lerp_parameters
def test_lerp5():
    """
    x, y: broadcast
    weight = 0.2
    """
    x = np.random.rand(4, 5)
    y = np.random.rand(1)
    w = 0.2
    res = x + w * (y - x)
    obj.run(res=res, x=x, y=y, weight=w)
