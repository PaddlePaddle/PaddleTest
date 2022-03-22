#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_logit
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def cal_logit(data, eps=None):
    """
    calculate logit
    """
    x = data.copy()
    shape = x.shape
    if eps:
        x[x < eps] = eps
        x[x > (1 - eps)] = 1 - eps
    res = []
    for item in x.flatten():
        if item < 0 or item > 1:
            res.append(np.nan)
        else:
            res.append(np.log(item / (1 - item)))
    return np.array(res).reshape(shape)


class TestLogit(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-3


obj = TestLogit(paddle.logit)


@pytest.mark.api_base_logit_vartype
def test_logit_base():
    """
    base
    """
    x = randtool("float", 0, 1, (4,))
    res = cal_logit(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_logit_parameters
def test_logit0():
    """
    x: 2d-tensor
    """
    obj.enable_backward = False
    x = randtool("float", -2, 2, (4, 2))
    res = cal_logit(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_logit_parameters
def test_logit1():
    """
    x: 3d-tensor
    """
    x = randtool("float", 0, 1, (4, 3, 2))
    res = cal_logit(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_logit_parameters
def test_logit2():
    """
    x: 4d-tensor
    """
    x = randtool("float", 0, 1, (4, 3, 2, 5))
    res = cal_logit(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_logit_parameters
def test_logit3():
    """
    x: 4d-tensor
    eps = 0.2
    """
    obj.enable_backward = True
    x = randtool("float", 0, 1, (4, 3, 2, 5))
    res = cal_logit(x, eps=0.2)
    obj.run(res=res, x=x, eps=0.2)
