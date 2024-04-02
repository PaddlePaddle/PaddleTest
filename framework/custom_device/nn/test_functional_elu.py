#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_functional_elu.py
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestFunctionalElu(APIBase):
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


obj = TestFunctionalElu(paddle.nn.functional.elu)


@pytest.mark.api_nn_elu_vartype
def test_elu_base():
    """
    base
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    alpha = 1
    res = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
    # print(res)
    obj.base(res=res, x=x)


@pytest.mark.api_nn_elu_parameters
def test_elu():
    """
    default
    """
    x = np.array([[-1, 6], [1, 15.6]])
    alpha = 0.2
    res = np.array([[-0.12642411, 6], [1, 15.6]])
    # print(res)
    obj.run(res=res, x=x, alpha=alpha)


@pytest.mark.api_nn_elu_parameters
def test_elu1():
    """
    alpha = 2
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    alpha = 2
    res = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
    # print(res)
    obj.run(res=res, x=x, alpha=alpha)


@pytest.mark.api_nn_elu_parameters
def test_elu2():
    """
    alpha = 0
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    alpha = 0
    res = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
    # print(res)
    obj.run(res=res, x=x, alpha=alpha)


# def test_elu3():
#     """
#     alpha = -1
#     """
#     x = randtool("float", -10, 10, [3, 3, 3])
#     alpha = -1
#     res = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
#     # print(res)
#     obj.run(res=res, x=x, alpha=alpha)
