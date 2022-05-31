#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_as_complex
"""
import numpy as np
import paddle
import pytest
from apibase import randtool
from apibase import APIBase


class TestAsComplex(APIBase):
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


obj = TestAsComplex(paddle.as_complex)


@pytest.mark.api_base_as_complex_vartype
def test_as_complex_base():
    """
    base
    """
    x = randtool("float", -4, 4, (3, 2))
    res = x[:, 0] + 1j * x[:, 1]
    obj.base(res=res, x=x)


@pytest.mark.api_base_as_complex_parameters
def test_as_complex1():
    """
    default
    """
    x = randtool("float", -4, 4, (9, 3, 2))
    res = x[:, :, 0] + 1j * x[:, :, 1]
    obj.run(res=res, x=x)


@pytest.mark.api_base_as_complex_parameters
def test_as_complex2():
    """
    x: 4d - tensor
    """
    x = randtool("float", -4, 4, (9, 2, 3, 2))
    res = x[:, :, :, 0] + 1j * x[:, :, :, 1]
    obj.run(res=res, x=x)
