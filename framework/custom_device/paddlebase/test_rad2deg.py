#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_rad2deg
"""

import sys
import os
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2


class TestRad2deg(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-3


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestRad2deg(paddle.rad2deg)


@skip_branch_is_2_2
@pytest.mark.api_base_rad2deg_vartype
def test_rad2deg_base():
    """
    base
    """
    x = np.random.randint(-4, 4, (10,))
    res = np.rad2deg(x)
    obj.base(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rad2deg_parameters
def test_rad2deg_0():
    """
    default
    """
    x = np.array([np.pi / 2])
    res = np.rad2deg(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rad2deg_parameters
def test_rad2deg_1():
    """
    x: 2-d tensor
    """
    x = np.random.rand(4, 4) * 10 - 5
    res = np.rad2deg(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rad2deg_parameters
def test_rad2deg_2():
    """
    x: 3-d tensor
    """
    x = np.random.rand(4, 4, 4) * 10 - 5
    res = np.rad2deg(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rad2deg_parameters
def test_rad2deg_3():
    """
    x: 3-d tensor
    """
    x = np.random.rand(4, 4, 4, 4) * 100 - 50
    res = np.rad2deg(x)
    obj.run(res=res, x=x)
