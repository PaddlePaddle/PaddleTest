#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_softshrink
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestSoftShrink(APIBase):
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


obj = TestSoftShrink(paddle.nn.functional.softshrink)


@pytest.mark.api_nn_softshrink_vartype
def test_softshrink_base():
    """
    base
    """
    x = np.array([-0.9, -0.2, 0.1, 0.8])
    threshold = 0.5
    res = []
    for i in x.flatten():
        if i > threshold:
            res.append(i - threshold)
        elif i < (-1 * threshold):
            res.append(i + threshold)
        else:
            res.append(0)
    res = np.array(res).reshape(x.shape)
    obj.base(res=res, x=x, threshold=threshold)


@pytest.mark.api_nn_softshrink_parameters
def test_softshrink():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    threshold = 5
    res = []
    for i in x.flatten():
        if i > threshold:
            res.append(i - threshold)
        elif i < (-1 * threshold):
            res.append(i + threshold)
        else:
            res.append(0)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x, threshold=threshold)


@pytest.mark.api_nn_softshrink_parameters
def test_softshrink1():
    """
    threshold = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    threshold = 0
    res = []
    for i in x.flatten():
        if i > threshold:
            res.append(i - threshold)
        elif i < (-1 * threshold):
            res.append(i + threshold)
        else:
            res.append(0)
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x, threshold=threshold)


@pytest.mark.api_nn_softshrink_exception
def test_softshrink2():
    """
    exception threshold = -3.3
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    threshold = -3.3
    res = []
    for i in x.flatten():
        if i > threshold:
            res.append(i - threshold)
        elif i < (-1 * threshold):
            res.append(i + threshold)
        else:
            res.append(0)
    res = np.array(res).reshape(x.shape)
    obj.exception(etype=ValueError, mode="python", x=x, threshold=threshold)
