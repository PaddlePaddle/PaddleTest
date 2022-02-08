#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cov
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCov(APIBase):
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
        # self.enable_backward = False
        # self.delta = 1e-3


obj = TestCov(paddle.linalg.cov)


@pytest.mark.api_linalg_cov_vartype
def test_cov_base():
    """
    base
    """
    x = randtool("float", -2, 2, (4, 2))
    res = np.cov(x)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_cov_parameters
def test_diff0():
    """
    default
    """
    x = randtool("float", -2, 2, (4, 12))
    res = np.cov(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_cov_parameters
def test_diff1():
    """
    rowvar: True
    """
    x = randtool("float", -4, 2, (4, 12))
    res = np.cov(x, rowvar=True)
    obj.run(res=res, x=x, rowvar=True)


@pytest.mark.api_linalg_cov_parameters
def test_diff2():
    """
    rowvar: True
    ddof: True
    """
    x = randtool("float", -4, 2, (4, 12))
    res = np.cov(x, rowvar=True, ddof=True)
    obj.run(res=res, x=x, rowvar=True, ddof=True)
