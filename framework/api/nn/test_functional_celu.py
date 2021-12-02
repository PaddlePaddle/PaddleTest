#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_functional_celu
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCELU(APIBase):
    """
    test CELU
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestCELU(paddle.nn.functional.celu)


def cal_celu_api(x, alpha=1.0):
    """
    calculate celu
    """
    min_value = alpha * (np.exp(x / alpha) - 1)
    r0 = np.where(min_value > 0, 0, min_value)
    r1 = np.where(x < 0, 0, x)
    r = r0 + r1
    return r


@pytest.mark.api_nn_CELU_vartype
def test_celu_base():
    """
    base
    """
    x = randtool("float", -2, 4, (2, 3, 4))
    res = cal_celu_api(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_CELU_parameters
def test_celu0():
    """
    default
    """
    x = randtool("float", -4, 3, (2, 4, 4))
    res = cal_celu_api(x)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_CELU_parameters
def test_celu1():
    """
    alpha = 0.2
    """
    x = randtool("float", -4, 3, (2, 4, 4))
    res = cal_celu_api(x, alpha=0.2)
    obj.run(res=res, x=x, alpha=0.2)


@pytest.mark.api_nn_CELU_parameters
def test_celu2():
    """
    alpha = -0.4
    """
    obj.data = 1e-4
    obj.enable_backward = False
    x = randtool("float", -4, 3, (2, 2))
    res = cal_celu_api(x, alpha=-0.4)
    obj.run(res=res, x=x, alpha=-0.4)
