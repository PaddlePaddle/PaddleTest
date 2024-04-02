#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test isnan
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestIsnan(APIBase):
    """
    test isnan
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


obj = TestIsnan(paddle.isnan)


@pytest.mark.api_base_isnan_vartype
def test_isnan_base():
    """
    base
    """
    x = np.array([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
    res = np.isnan(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_isnan_parameters
def test_isnan1():
    """
    x = np.nan
    """
    x = np.array([np.nan, np.inf, np.NINF, -np.inf, -np.nan])
    res = np.isnan(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isnan_parameters
def test_isnan2():
    """
    x =np.zeros([2])
    """
    x = np.zeros([2])
    res = np.isnan(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isnan_parameters
def test_isnan4():
    """
    x =np.ones([2])
    """
    x = np.ones([2])
    res = np.isnan(x)
    obj.run(res=res, x=x)


# exception case
# def test_isnan5():
#     """
#     x = np.array([]),[]?None,icafe
#     """
#     x = np.array([])
#     res = np.isnan(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_isnan_exception
def test_isnan6():
    """
    x = np.array(['inf']),c++error,icafe
    """
    x = np.array(["inf"])
    obj.exception(x=x, mode="c", etype="InvalidArgumentError")


@pytest.mark.api_base_isnan_parameters
def test_isnan7():
    """
    x =np.array([float('-nan'), float('-inf'), float('nan')])
    """
    x = np.array([float("-nan"), float("-inf"), float("nan")])
    res = np.isnan(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isnan_parameters
def test_isnan8():
    """
    x =np.array([float('-inf'), float('-inf'), float('inf')])
    """
    x = np.array([float("-inf"), float("-inf"), float("inf")])
    res = np.isnan(x)
    obj.run(res=res, x=x)
