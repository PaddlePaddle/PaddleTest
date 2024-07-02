#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_selu
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestSelu(APIBase):
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


obj = TestSelu(paddle.nn.functional.selu)


@pytest.mark.api_nn_selu_vartype
def test_selu_base():
    """
    base
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = np.array([[0.1, 1], [2, 3]])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_selu_parameters
def test_selu():
    """
    default
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = randtool("float", -10, 10, [3, 3, 3])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x)


@pytest.mark.api_nn_selu_parameters
def test_selu1():
    """
    alpha = 0
    """
    alpha = 0
    scale = 1.0507009873554804934193349852946
    x = randtool("float", -10, 10, [3, 3, 3])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    obj.run(res=res, x=x, alpha=alpha, scale=scale)


@pytest.mark.api_nn_selu_exception
def test_selu2():
    """
    alpha = -3.333 bug
    """
    alpha = -3.333
    scale = 1.0507009873554804934193349852946
    x = randtool("float", -10, 10, [3, 3, 3])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    obj.exception(etype=ValueError, mode="python", x=x, alpha=alpha, scale=scale)


@pytest.mark.api_nn_selu_exception
def test_selu3():
    """
    scale = 0
    """
    alpha = 3.3
    scale = 0
    x = randtool("float", -10, 10, [3, 3, 3])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    obj.exception(etype=ValueError, mode="python", x=x, alpha=alpha, scale=scale)


@pytest.mark.api_nn_selu_exception
def test_selu4():
    """
    scale = -3.33
    """
    alpha = 1
    scale = -3.33
    x = randtool("float", -10, 10, [3, 3, 3])
    x = np.array([-1])
    res = []
    for i in x.flatten():
        if i > 0:
            res.append(scale * i)
        else:
            res.append(scale * (alpha * np.exp(i) - alpha))
    res = np.array(res).reshape(x.shape)
    # res = scale * (np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1)))
    obj.exception(etype=ValueError, mode="python", x=x, alpha=alpha, scale=scale)
