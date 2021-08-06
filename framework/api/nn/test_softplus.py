#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_softplus
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestSoftplus(APIBase):
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


obj = TestSoftplus(paddle.nn.Softplus)


@pytest.mark.api_nn_SOFTPLUS_vartype
def test_softplus_base():
    """
    base
    """
    x = np.array([-0.4, -0.2, 0.1, 0.3])
    beta = 1
    threshold = 15
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.base(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = 2
    threshold = 20
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus1():
    """
    beta = 0.000001
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = 0.000001
    threshold = 20
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus2():
    """
    beta = -0.000001
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = -0.000001
    threshold = 20
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus3():
    """
    beta = -3
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = -3
    threshold = 20
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus4():
    """
    beta = 3  threshold = 5
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = 3
    threshold = 5
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)


@pytest.mark.api_nn_SOFTPLUS_parameters
def test_softplus5():
    """
    beta = 3  threshold = -5
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    beta = 3
    threshold = -5
    x_beta = beta * x
    res = np.select([x_beta <= threshold, x_beta > threshold], [np.log(1 + np.exp(x_beta)) / beta, x])
    obj.run(res=res, data=x, beta=beta, threshold=threshold)
