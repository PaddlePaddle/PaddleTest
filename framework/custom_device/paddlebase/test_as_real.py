#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_as_real
"""
import numpy as np
import paddle
import pytest
from apibase import randtool
from apibase import APIBase


class TestAsReal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.complex64, np.complex128]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-5


obj = TestAsReal(paddle.as_real)


@pytest.mark.api_base_as_real_vartype
def test_as_real_base():
    """
    base
    """
    x0 = randtool("float", -4, 4, (3, 2))
    x1 = x0[:, 0] + 1j * x0[:, 1]
    obj.base(res=x0, x=x1)


@pytest.mark.api_base_as_real_parameters
def test_as_real1():
    """
    default
    """
    x = randtool("float", -4, 4, (9, 3, 2))
    res = x[:, :, 0] + 1j * x[:, :, 1]
    obj.base(res=x, x=res)


@pytest.mark.api_base_as_real_parameters
def test_as_real2():
    """
    x: 4d - tensor
    """
    x = randtool("float", -4, 4, (9, 2, 3, 2))
    res = x[:, :, :, 0] + 1j * x[:, :, :, 1]
    obj.base(res=x, x=res)
