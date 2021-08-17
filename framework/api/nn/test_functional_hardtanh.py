#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_hardtanh
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalHardtanh(APIBase):
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


obj = TestFunctionalHardtanh(paddle.nn.functional.hardtanh)


@pytest.mark.api_nn_hardtanh_vartype
def test_hardtanh_base():
    """
    base
    """
    x = np.array([-1.5, 0.3, 2.5])
    res = np.array([-1, 0.3, 1])
    obj.base(res=res, x=x)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = 3.5
    paddle_min = 2.3
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh1():
    """
    max = 1.3 min = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = 1.3
    paddle_min = 0
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh2():
    """
    max = 0 min = -3.4
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = 0
    paddle_min = -3.4
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh3():
    """
    max = 1 min = 1
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = 1
    paddle_min = 1
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh4():
    """
    max = 0 min = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = 0
    paddle_min = 0
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh5():
    """
    max = -3.2 min = -3.2
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = -3.2
    paddle_min = -3.2
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_parameters
def test_hardtanh6():
    """
    max < min max=-3.3 min=3.3
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = -3.3
    paddle_min = 3.3
    res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.run(res=res, x=x, max=paddle_max, min=paddle_min)


@pytest.mark.api_nn_hardtanh_exception
def test_hardtanh7():
    """
    exception max is a string
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    paddle_max = "-3.3"
    paddle_min = 3.3
    # res = np.minimum(np.maximum(x, paddle_min), paddle_max)
    obj.exception(etype="InvalidArgumentError", x=x, max=paddle_max, min=paddle_min)
