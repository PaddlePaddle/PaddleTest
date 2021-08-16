#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_relu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalRelu(APIBase):
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


obj = TestFunctionalRelu(paddle.nn.functional.relu)


@pytest.mark.api_nn_relu_vartype
def test_relu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = np.maximum(0, x)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_relu_parameters
def test_relu():
    """
    default
    """
    x = randtool("float", -10, 10, [10, 10, 10])
    res = np.maximum(0, x)
    obj.run(res=res, x=x)
