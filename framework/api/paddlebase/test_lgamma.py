#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lgamma
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestLgamma(APIBase):
    """
    test lgamma
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.delta = 1e-5
        self.rtol = 1e-5
        self.enable_backward = False


obj = TestLgamma(paddle.lgamma)
obj1 = TestLgamma(paddle.lgamma)
obj1.delta = 5 * 1e-2
obj1.enable_backward = True


def compute_lgamma(tensor):
    """
    numeric compute
    """
    shape = tensor.shape
    res = []
    for i in tensor.flatten():
        res.append(math.lgamma(i))
    res = np.array(res).reshape(shape)
    return res


@pytest.mark.api_base_lgamma_vartype
def test_lgamma_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6])
    res = compute_lgamma(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_lgamma_vartype
def test_lgamma_base_backward():
    """
    test backward delta = 0.05
    """
    x = randtool("float", -5, 5, [3, 3])
    print(x)
    res = compute_lgamma(x)
    obj1.base(res=res, x=x)


@pytest.mark.api_base_lgamma_parameters
def test_lgamma():
    """
    base float
    """
    x = randtool("float", -5, 5, [6, 6, 6])
    res = compute_lgamma(x)
    obj.run(res=res, x=x)
