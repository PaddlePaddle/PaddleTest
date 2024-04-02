#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cholesky_solve
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCholeskySolve(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False
        self.delta = 1e-4


obj = TestCholeskySolve(paddle.linalg.cholesky_solve)


def cal_cholesky_solve(x, y, upper=False):
    """
    calculate cholesky_slove
    """
    shape = y.shape
    shape1 = x.shape
    batchsize = np.product(shape) // (shape[-2] * shape[-1])
    y_t = y.reshape(-1, shape[-2], shape[-1])
    x_t = x.reshape(-1, shape1[-2], shape1[-1])
    tmp = []
    for i in range(batchsize):
        c_u = 0
        if not upper:
            u = np.tril(y_t[i])
            c_u = np.dot(u, u.T)
        if upper:
            u = np.triu(y_t[i])
            c_u = np.dot(u.T, u)
        tmp.append(np.dot(np.linalg.inv(c_u), x_t[i]))
    return np.array(tmp).reshape(shape1)


@pytest.mark.api_linalg_cholesky_solve_vartype
def test_cholesky_solve_base():
    """
    base
    """
    x = randtool("float", -4, 4, (4, 1))
    y = randtool("float", -2, 2, (4, 4))
    res = cal_cholesky_solve(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_linalg_cholesky_solve_parameters
def test_cholesky_solve0():
    """
    default
    """
    x = randtool("float", -4, 4, (4, 3))
    y = randtool("float", -2, 2, (4, 4))
    res = cal_cholesky_solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_cholesky_solve_parameters
def test_cholesky_solve1():
    """
    x,y : 3d-tensor
    """
    x = randtool("float", -4, 4, (5, 4, 3))
    y = randtool("float", -2, 2, (5, 4, 4))
    res = cal_cholesky_solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_cholesky_solve_parameters
def test_cholesky_solve2():
    """
    x,y : 4d-tensor
    """
    x = randtool("float", -4, 4, (5, 2, 4, 3))
    y = randtool("float", -2, 2, (5, 2, 4, 4))
    res = cal_cholesky_solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_cholesky_solve_parameters
def test_cholesky_solve3():
    """
    x,y : 4d-tensor
    upper=True
    """
    x = randtool("float", -4, 4, (5, 2, 4, 3))
    y = randtool("float", -2, 2, (5, 2, 4, 4))
    res = cal_cholesky_solve(x, y, upper=True)
    obj.run(res=res, x=x, y=y, upper=True)
