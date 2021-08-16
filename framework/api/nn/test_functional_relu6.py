#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_relu6"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalRelu6(APIBase):
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


obj = TestFunctionalRelu6(paddle.nn.functional.relu6)


@pytest.mark.api_nn_relu6_vartype
def test_relu6_base():
    """
    base
    """
    x = np.array([-1, 0.3, 6.5])
    res = np.minimum(np.maximum(0, x), 6)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_relu6_parameters
def test_relu6():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = np.minimum(np.maximum(0, x), 6)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_relu6_parameters
def test_relu61():
    """
    x = np.array([6, 6, 6, 6]
    """
    x = np.array([6, 6, 6, 6])
    res = np.minimum(np.maximum(0, x), 6)
    obj.run(res=res, x=x)
