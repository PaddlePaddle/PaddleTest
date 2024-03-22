#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test real
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestReal(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.complex64, np.complex128]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestReal(paddle.real)


@pytest.mark.api_base_real_vartype
def test_real_base():
    """
    base
    Returns:

    """
    x = np.random.random((20, 10)) + 1j * np.random.random((20, 10))
    res = np.real(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_real_parameters
def test_real():
    """
    base
    Returns:

    """
    x = np.random.random((1, 10)) + 1j * np.random.random((1, 10))
    res = np.real(x)
    obj.run(res=res, x=x)
