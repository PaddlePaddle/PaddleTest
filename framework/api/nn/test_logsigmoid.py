#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_logsigmoid
"""
from apibase import APIBase
from apibase import randtool
from apibase import sigmoid
import paddle
import pytest
import numpy as np


class TestLogSigmoid(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestLogSigmoid(paddle.nn.LogSigmoid)


@pytest.mark.api_nn_LogSigmoid_vartype
def test_logsigmoid_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    # x = np.array([1, 2, 3, 4])
    res = np.log(sigmoid(x))
    obj.base(res=res, data=x)


@pytest.mark.api_nn_LogSigmoid_parameters
def test_logsigmoid():
    """
    default
    """
    x = randtool("float", -10, 10, [10, 10, 10])
    res = np.log(sigmoid(x))
    obj.base(res=res, data=x)
