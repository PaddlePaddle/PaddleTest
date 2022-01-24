#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_isclose
"""

import sys
from apibase import APIBase

import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2
import os


class TestIsClose(APIBase):
    """
    test isclose
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestIsClose(paddle.isclose)


@skip_branch_is_2_2
@pytest.mark.api_base_isclose_vartype
def test_isclose_base():
    """
    base
    """
    x = np.array([10000.0, 1e-07])
    y = np.array([10000.1, 1e-08])
    res = np.isclose(x, y)
    obj.base(res=res, x=x, y=y)


@skip_branch_is_2_2
@pytest.mark.api_base_isclose_parameters
def test_isclose0():
    """
    x, y: multiple dimension
    """
    x = np.random.rand(3, 4, 5) * 100
    y = x + 0.001
    res = np.isclose(x, y)
    obj.run(res=res, x=x, y=y)


@skip_branch_is_2_2
@pytest.mark.api_base_isclose_parameters
def test_isclose1():
    """
     equal_nan=True
    """
    x = np.array([10000.0, 1e-07, np.NAN, 1.0, 3.0, 0.0])
    y = np.array([10000.01, 1e-06, np.NAN, np.NAN, 3.0, np.NAN])
    res = np.isclose(x, y)
    obj.run(res=res, x=x, y=y)
