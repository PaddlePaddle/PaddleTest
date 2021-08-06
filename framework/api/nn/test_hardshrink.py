#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_hardshrink
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestNNHardshrink(APIBase):
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


obj = TestNNHardshrink(paddle.nn.Hardshrink)


@pytest.mark.api_nn_HARDSHRINK_vartype
def test_hardshrink_base():
    """
    base
    """
    x = np.array([-1, 0.3, 2.5])
    res = np.array([-1, 0, 2.5])
    obj.base(res=res, data=x)


@pytest.mark.api_nn_HARDSHRINK_parameters
def test_hardshrink():
    """
    default
    """
    x = np.array([-1, 0.3, 2.5])
    res = np.array([-1, 0, 2.5])
    obj.run(res=res, data=x)


@pytest.mark.api_nn_HARDSHRINK_parameters
def test_hardshrink1():
    """
    threshold = 0
    """
    x = np.array([-1, 0.3, 2.5])
    threshold = 0
    res = np.array([-1, 0.3, 2.5])
    obj.run(res=res, data=x, threshold=threshold)


@pytest.mark.api_nn_HARDSHRINK_parameters
def test_hardshrink2():
    """
    threshold = 0 x contains 0.01
    """
    x = np.array([-1, -0.01, 2.5])
    threshold = 0
    res = np.array([-1, -0.01, 2.5])
    obj.run(res=res, data=x, threshold=threshold)


@pytest.mark.api_nn_HARDSHRINK_vartype
def test_hardshrink3():
    """
    threshold = -1
    """
    x = np.array([-1, -0.01, 2.5])
    threshold = -1
    res = np.array([-1, -0.01, 2.5])
    obj.base(res=res, data=x, threshold=threshold)


@pytest.mark.api_nn_HARDSHRINK_exception
def test_hardshrink4():
    """
    threshold = "1"
    """
    x = np.array([-1, -0.01, 2.5])
    threshold = "1"
    # res = np.array([-1, -0.01, 2.5])
    obj.exception(etype="InvalidArgumentError", data=x, threshold=threshold)
