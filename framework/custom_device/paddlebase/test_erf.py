#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test erf
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestErf(APIBase):
    """
    test erf
    """

    def hook(self):
        """
        implement
        """
        # self.types = [np.float16, np.float32, np.float64]
        self.types = [np.float32, np.float64]
        self.enable_backward = True


obj = TestErf(paddle.erf)


@pytest.mark.api_base_erf_vartype
def test_erf_base():
    """
    base
    """
    x = np.array([0.9, 0.2, 3.1, 0.3])
    res = np.array([0.79690821, 0.22270259, 0.99998835, 0.32862676])
    obj.base(res=res, x=x)


@pytest.mark.api_base_erf_parameters
def test_erf1():
    """
    input is multi-dim
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = np.array([[0.74210096, 0.42839236], [0.67780119, 0.79690821]])
    obj.base(res=res, x=x)


@pytest.mark.api_base_erf_parameters
def test_erf2():
    """
    input has negtive num
    """
    x = np.array([-0.4, -0.2, 0.1, 0.3])
    res = np.array([-0.42839236, -0.22270259, 0.11246292, 0.32862676])
    obj.base(res=res, x=x)


class TestErf1(APIBase):
    """
    test erf
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        self.enable_backward = True
        self.debug = True


obj1 = TestErf1(paddle.erf)


# def test_erf3():
#     """
#     x is float16
#     Cannot support float16 in cpu, it's a common problem.
#     """
#     x = np.array([-0.4, -0.2, 0.1, 0.3])
#     res = np.array([-0.4282, -0.2225, 0.1124, 0.3286])
#     obj1.base(res=res, x=x)
