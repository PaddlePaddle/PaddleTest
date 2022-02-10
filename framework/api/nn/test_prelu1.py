#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_prelu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


def cal_prelu(data, weight):
    """
    calculate prelu
    """
    x = data.copy()
    x[x <= 0] = weight * x[x <= 0]
    return x


class TestPRelu(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = True


obj = TestPRelu(paddle.nn.PReLU)


@pytest.mark.api_nn_PReLU_vartype
def test_prelu_base():
    """
    base
    """
    data = randtool("float", -10, 10, [2, 3, 3, 3])
    init = 0.25
    res = cal_prelu(data, init)
    obj.base(res=res, data=data)


@pytest.mark.api_nn_PReLU_parameters
def test_prelu_float64():
    """
    base float64
    """
    obj.types = [np.float64]

    paddle.set_default_dtype("float64")
    data = randtool("float", -10, 10, [2, 3, 3, 3])
    init = 0.25
    res = cal_prelu(data, init)
    obj.run(res=res, data=data)


@pytest.mark.api_nn_PReLU_vartype
def test_prelu0():
    """
    data_format = NHWC
    """
    data = randtool("float", -10, 10, [2, 4, 5, 6])
    init = 0.25
    res = cal_prelu(data, init)
    obj.base(res=res, data=data, data_format="NHWC")
