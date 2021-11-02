#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cholesky
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCholesky(APIBase):
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
        # self.enable_backward = False
        self.delta = 1e-1


obj = TestCholesky(paddle.cholesky)


@pytest.mark.api_linalg_cholesky_vartype
def test_cholesky_base():
    """
    base
    """
    x = randtool("float", 1, 10, [4, 4])
    res = np.tril(x)
    x = np.dot(res, np.transpose(res, (1, 0)))
    # print(res)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_cholesky_parameters
def test_cholesky0():
    """
    multidimensional
    """
    x = randtool("float", 1, 10, [4, 4])
    res = np.tril(x)
    x = np.dot(res, np.transpose(res, (1, 0)))
    res = res.reshape((1, 1, 4, 4))
    x = x.reshape((1, 1, 4, 4))
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_cholesky_parameters
def test_cholesky1():
    """
    upper = True
    """
    x = randtool("float", 1, 10, [4, 4])
    res = np.triu(x)
    x = np.dot(np.transpose(res, (1, 0)), res)
    obj.run(res=res, x=x, upper=True)
