#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test isfinite
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestIsfinite(APIBase):
    """
    test isfinite
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16, np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        self.no_grad_var = {"x"}
        # enable check grad
        self.enable_backward = False


obj = TestIsfinite(paddle.isfinite)


@pytest.mark.api_base_isfinite_vartype
def test_isfinite_base():
    """
    base
    """
    x = np.array([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
    res = np.isfinite(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_isfinite_parameters
def test_isfinite1():
    """
    x = np.nan
    """
    x = np.array([np.nan, np.inf, np.NINF, -np.inf, -np.nan])
    res = np.isfinite(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isfinite_parameters
def test_isfinite2():
    """
    x =np.zeros([2])
    """
    x = np.zeros([2])
    res = np.isfinite(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isfinite_parameters
def test_isfinite4():
    """
    x =np.ones([2])
    """
    x = np.ones([2])
    res = np.isfinite(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isfinite_parameters
def test_isfinite5():
    """
    x =np.array([float('-inf'), float('-inf'), float('inf')])
    """
    x = np.array([float('-inf'), float('-inf'), float('inf')])
    res = np.isfinite(x)
    obj.run(res=res, x=x)