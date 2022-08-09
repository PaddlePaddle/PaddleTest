#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_complex
"""
import numpy as np
import paddle
import pytest
from apibase import randtool
from apibase import APIBase


class TestLstsq(APIBase):
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
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestLstsq(paddle.complex)


@pytest.mark.api_base_lstsq_vartype
def test_lstsq_base():
    """
    base
    """
    x = np.array([[1, 3], [3, 2], [5, 6.0]])
    y = np.array([[1, 3], [3, 2], [5, 6.0]])
    res = x + y * 1j
    obj.base(res=res, real=x, imag=y)


@pytest.mark.api_base_lstsq_parameters
def test_lstsq1():
    """
    default
    """
    x = randtool("float", -4, 4, (9,))
    y = randtool("float", -4, 4, (9,))
    res = x + y * 1j
    obj.base(res=res, real=x, imag=y)


@pytest.mark.api_base_lstsq_parameters
def test_lstsq2():
    """
    default
    """
    x = randtool("float", -4, 4, (9, 2, 3))
    y = randtool("float", -4, 4, (9, 2, 3))
    res = x + y * 1j
    obj.base(res=res, real=x, imag=y)
