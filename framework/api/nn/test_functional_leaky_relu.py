#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_leaky_relu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestNNLeakyReLU(APIBase):
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


obj = TestNNLeakyReLU(paddle.nn.functional.leaky_relu)


@pytest.mark.api_nn_leaky_relu_vartype
def test_nn_leakyrelu_base():
    """
    base
    """
    data = np.array([-2, 0.1, 1])
    res = np.array([-0.02, 0.1, 1])
    obj.base(res=res, x=data)


@pytest.mark.api_nn_leaky_relu_parameters
def test_nn_leakyrelu():
    """
    default
    """
    data = np.array([-2, -7, 0.1, 1, 300])
    res = np.array([-0.02, -0.07, 0.1, 1, 300])
    obj.run(res=res, x=data)


@pytest.mark.api_nn_leaky_relu_parameters
def test_nn_leakyrelu1():
    """
    default negative_slope=0
    """
    data = np.array([-2, -7, 0.1, 1, 300])
    res = np.array([0, 0, 0.1, 1, 300])
    negative_slope = 0
    obj.run(res=res, x=data, negative_slope=negative_slope)


@pytest.mark.api_nn_leaky_relu_parameters
def test_nn_leakyrelu2():
    """
    default negative_slope=-100
    """
    data = np.array([-2, -7, 0.1, 1, 300])
    res = np.array([200, 700, 0.1, 1, 300])
    negative_slope = -100
    obj.run(res=res, x=data, negative_slope=negative_slope)
