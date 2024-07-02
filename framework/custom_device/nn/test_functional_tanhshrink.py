#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_tanhshrink
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalTanhshrink(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestFunctionalTanhshrink(paddle.nn.functional.tanhshrink)


@pytest.mark.api_nn_tanhshrink_vartype
def test_tanhshrink_base():
    """
    base
    """
    x = np.array([-0.4, -0.2, 0.1, 0.3])
    res = x - np.tanh(x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_tanhshrink_parameters
def test_tanhshrink():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = x - np.tanh(x)
    obj.run(res=res, x=x)
