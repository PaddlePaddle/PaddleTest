#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_relu
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRelu(APIBase):
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


obj = TestRelu(paddle.nn.ReLU)


@pytest.mark.api_nn_ReLU_vartype
def test_relu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 10, 3, 3])
    res = np.maximum(0, x)
    # print(res)
    obj.base(res=res, data=x)


@pytest.mark.api_nn_ReLU_parameters
def test_relu():
    """
    default
    """
    x = np.array([[-1, 4], [1, 15.6]])
    res = np.array([[0, 4], [1, 15.6]])
    # print(res)
    obj.run(res=res, data=x)
