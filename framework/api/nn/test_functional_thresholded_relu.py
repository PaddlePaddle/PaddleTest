#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_thresholded_relu
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalThresholdedReLU(APIBase):
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


obj = TestFunctionalThresholdedReLU(paddle.nn.functional.thresholded_relu)


@pytest.mark.api_nn_thresholded_relu_vartype
def test_thresholded_relu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [100, 4, 3, 3])
    # x = np.random.rand(1, 2, 2, 2)
    # print(x)
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] > 1.0:
            res.append(x.flatten()[i])
        else:
            res.append(0)
    # print(res)
    res = np.array(res).reshape(x.shape)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_thresholded_relu_parameters
def test_thresholded_relu():
    """
     default: threshold = 1.0
    """
    x = randtool("float", -10, 10, [100, 4, 3, 3])
    # x = np.random.rand(1, 2, 2, 2)
    # print(x)
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] > 1.0:
            res.append(x.flatten()[i])
        else:
            res.append(0)
    # print(res)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_thresholded_relu_parameters
def test_thresholded_relu1():
    """
    threshold = -1
    """

    x = randtool("float", -10, 1, [10, 1, 4, 3])
    # x = np.random.rand(1, 2, 2, 2)
    # print(x)
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] > -1.0:
            res.append(x.flatten()[i])
        else:
            res.append(0)
    # print(res)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x, threshold=-1)


@pytest.mark.api_nn_thresholded_relu_parameters
def test_thresholded_relu2():
    """
    threshold = 0
    """

    x = randtool("float", -10, 1, [10, 1, 4, 3])
    # x = np.random.rand(1, 2, 2, 2)
    # print(x)
    res = []
    for i in range(len(x.flatten())):
        if x.flatten()[i] > 0:
            res.append(x.flatten()[i])
        else:
            res.append(0)
    # print(res)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x, threshold=0)
