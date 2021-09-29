#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_solve
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestSolve(APIBase):
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


obj = TestSolve(paddle.linalg.solve)


@pytest.mark.api_linalg_solve_vartype
def test_solve_base():
    """
    base
    """
    x = randtool("float", -2, 4, (4, 4))
    y = randtool("float", -1, 1, (4,))
    res = np.linalg.solve(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_linalg_solve_parameters
def test_solve0():
    """
    default
    """
    x = randtool("float", -2, 4, (14, 14))
    y = randtool("float", -1, 1, (14,))
    res = np.linalg.solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_solve_parameters
def test_solve1():
    """
    y: multi_dim
    """
    x = randtool("float", -2, 4, (14, 14))
    y = randtool("float", -1, 1, (14, 2))
    res = np.linalg.solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_solve_parameters
def test_solve2():
    """
    x: batch_size != 1
    y: multi_dim
    """
    x = randtool("float", -2, 4, (4, 14, 14))
    y = randtool("float", -1, 1, (4, 14, 2))
    res = np.linalg.solve(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_linalg_solve_exception
def test_solve3():
    """
    x: not Invertable
    """
    x = np.ones((4, 4), dtype="float32")
    y = randtool("float", -1, 1, (4,))
    obj.exception(ValueError, mode="python", x=x, y=y)
