#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_triangular_solve
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestTriangularSolve(APIBase):
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


obj = TestTriangularSolve(paddle.linalg.triangular_solve)


def cal_traingular_solve(A, b, upper=True, transpose=False, unitriangular=False):
    """
    calculate traingular_solve api
    """
    A = np.triu(A)
    if upper is False:
        A = np.diag(np.diag(A))
    if transpose is True:
        A = np.transpose(A)
    if unitriangular is True:
        np.fill_diagonal(A, 1)
    solution = np.linalg.solve(A, b)

    return solution


@pytest.mark.api_linalg_triangular_solve_vartype
def test_triangular_solve_base():
    """
    base
    """

    x = np.random.rand(3, 3)
    x = np.triu(x)
    b = np.random.rand(3, 1)
    res = cal_traingular_solve(x, b)
    obj.base(res=res, x=x, y=b)


@pytest.mark.api_linalg_triangular_solve_parameters
def test_triangular_solve0():
    """
    default
    """
    obj.enable_backward = False
    x = np.random.rand(4, 4)
    x = np.triu(x)
    b = np.random.rand(4, 1)
    res = cal_traingular_solve(x, b)
    obj.run(res=res, x=x, y=b)


@pytest.mark.api_linalg_triangular_solve_parameters
def test_triangular_solve1():
    """
    y shape[1] > 1
    """
    obj.enable_backward = False
    x = np.random.rand(4, 4)
    x = np.triu(x)
    b = np.random.rand(4, 4)
    res = cal_traingular_solve(x, b)
    obj.run(res=res, x=x, y=b)


@pytest.mark.api_linalg_triangular_solve_parameters
def test_triangular_solve2():
    """
    y shape[1] > 1
    upper = False
    """
    obj.enable_backward = False
    x = np.random.rand(4, 4)
    x = np.triu(x)
    b = np.random.rand(4, 4)
    res = cal_traingular_solve(x, b, upper=False)
    obj.run(res=res, x=x, y=b, upper=False)


@pytest.mark.api_linalg_triangular_solve_parameters
def test_triangular_solve3():
    """
    y shape[1] > 1
    upper = False
    transpose = True
    """
    obj.enable_backward = False
    x = np.random.rand(4, 4)
    x = np.triu(x)
    b = np.random.rand(4, 4)
    res = cal_traingular_solve(x, b, upper=False, transpose=True)
    obj.run(res=res, x=x, y=b, upper=False, transpose=True)


@pytest.mark.api_linalg_triangular_solve_parameters
def test_triangular_solve4():
    """
    y shape[1] > 1
    upper = False
    transpose = True
    unitriangular=True
    """
    obj.enable_backward = False
    x = np.random.rand(4, 4)
    x = np.triu(x)
    b = np.random.rand(4, 4)
    res = cal_traingular_solve(x, b, upper=False, transpose=True, unitriangular=True)
    obj.run(res=res, x=x, y=b, upper=False, transpose=True, unitriangular=True)
