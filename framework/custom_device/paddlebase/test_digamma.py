#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test digamma
"""
import math

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np
from scipy.special import psi


class TestDigamma(APIBase):
    """
    test digamma
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestDigamma(paddle.digamma)
# test backward
obj1 = TestDigamma(paddle.digamma)
obj1.delta = 1
obj1.enable_backward = True


@pytest.mark.api_base_digamma_vartype
def test_digamma_base():
    """
    base float
    """
    x = randtool("float", -5, 5, [3, 3])
    res = psi(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_digamma_vartype
def test_digamma_base1():
    """
    backward
    """
    x = randtool("float", -5, 5, [3, 3])
    res = psi(x)
    obj1.base(res=res, x=x)


@pytest.mark.api_base_digamma_parameters
def test_digamma():
    """
    base float 3-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6])
    res = psi(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_digamma_parameters
def test_digamma1():
    """
    base float 4-D tensor
    """
    x = randtool("float", -5, 5, [6, 6, 6, 6])
    res = psi(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_digamma_parameters
def test_digamma2():
    """
    base float 5-D tensor
    """
    x = randtool("float", -5, 5, [3, 6, 6, 6, 6])
    res = psi(x)
    obj.run(res=res, x=x)
