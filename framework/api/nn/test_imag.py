#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test imag
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestImag(APIBase):
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


obj = TestImag(paddle.imag)


@pytest.mark.api_base_imag_vartype
def test_real_base():
    """
    base
    Returns:

    """
    x = np.random.random(
                (20, 10)) + 1j * np.random.random(
                    (20, 10))
    res = np.imag(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_imag_parameters
def test_real():
    """
    base
    Returns:

    """
    x = np.random.random(
                (1, 10)) + 1j * np.random.random(
                    (1, 10))
    res = np.imag(x)
    obj.run(res=res, x=x)
