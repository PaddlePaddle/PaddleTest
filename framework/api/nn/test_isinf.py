#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test isinf
"""
from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestIsinf(APIBase):
    """
    test isinf
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


obj = TestIsinf(paddle.isinf)


@pytest.mark.api_base_isinf_vartype
def test_isinf_base():
    """
    base
    """
    x = np.array([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
    res = np.isinf(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_isinf_parameters
def test_isinf1():
    """
    x = np.nan
    """
    x = np.array([np.nan, np.inf, np.NINF, -np.inf, -np.nan])
    res = np.isinf(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isinf_parameters
def test_isinf2():
    """
    x =np.zeros([2])
    """
    x = np.zeros([2])
    res = np.isinf(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_isinf_parameters
def test_isinf4():
    """
    x =np.ones([2])
    """
    x = np.ones([2])
    res = np.isinf(x)
    obj.run(res=res, x=x)


# exception case
# def test_isinf5():
#     """
#     x = np.array([]),[]?None,icafe
#     """
#     x = np.array([])
#     res = np.isinf(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_isinf_exception
def test_isinf6():
    """
    x = np.array(['inf']),c++error,icafe
    """
    x = np.array(['inf'])
    obj.exception(x=x, mode='c', etype='InvalidArgumentError')