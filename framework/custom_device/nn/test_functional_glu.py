#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_glu
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestGlu(APIBase):
    """
    test glu
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestGlu(paddle.nn.functional.glu)


def cal_glu(x, axis=-1):
    """
    calculate glu
    """
    a1, a2 = np.split(x, 2, axis=axis)
    a2 = 1 / (1 + np.exp(-a2))
    return a1 * a2


@pytest.mark.api_nn_glu_vartype
def test_glu_base():
    """
    base
    """
    x = np.random.rand(2, 4)
    res = cal_glu(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_glu_parameters
def test_glu0():
    """
    default
    """
    x = np.random.rand(10)
    res = cal_glu(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_glu_parameters
def test_glu1():
    """
    x: 3d-tensor
    """
    x = np.random.rand(10, 2, 8)
    res = cal_glu(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_glu_parameters
def test_glu2():
    """
    x: 4d-tensor
    """
    x = np.random.rand(4, 6, 2, 8)
    res = cal_glu(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_glu_parameters
def test_glu3():
    """
    x: 4d-tensor
    axis = 0
    """
    x = np.random.rand(4, 6, 2, 8)
    res = cal_glu(x, axis=0)
    obj.base(res=res, x=x, axis=0)


@pytest.mark.api_nn_glu_parameters
def test_glu4():
    """
    x: 4d-tensor
    axis = 2
    """
    x = np.random.rand(4, 6, 2, 8)
    res = cal_glu(x, axis=2)
    obj.base(res=res, x=x, axis=2)
