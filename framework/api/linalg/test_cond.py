#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_cond
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestCond(APIBase):
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
        self.enable_backward = False


obj = TestCond(paddle.linalg.cond)


@pytest.mark.api_linalg_cond_vartype
def test_cond_base():
    """
    base
    """
    x = np.array([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
    res = np.array([np.linalg.cond(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_cond_parameters
def test_cond0():
    """
    m!=n, p=2
    """
    x = randtool("float", -2, 4, [3, 4])
    # x = np.random.rand(3, 4)
    res = np.array([np.linalg.cond(x)])
    # print(res)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_cond_parameters
def test_cond1():
    """
    m!=n, p=-2
    """
    x = randtool("float", -2, 4, [6, 4])
    res = np.array([np.linalg.cond(x, p=-2)])
    obj.run(res=res, x=x, p=-2)


@pytest.mark.api_linalg_cond_parameters
def test_cond2():
    """
    m!=n, p=2
    x: multiple dimension
    """
    x = randtool("float", -20, 40, [6, 2, 4, 3, 4])
    res = np.array(np.linalg.cond(x))
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_cond_parameters
def test_cond3():
    """
    x: n*n； p=2
    """
    x = randtool("float", -20, 40, [4, 4])
    res = np.array([np.linalg.cond(x)])
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_cond_parameters
def test_cond4():
    """
    x: n*n； p=-2
    """
    x = randtool("float", -2, 40, [4, 4])
    res = np.array([np.linalg.cond(x, p=-2)])
    obj.run(res=res, x=x, p=-2)


@pytest.mark.api_linalg_cond_parameters
def test_cond5():
    """
    x: n*n； p=-2
    """
    x = randtool("float", -2, 40, [4, 4])
    res = np.array([np.linalg.cond(x, p=-2)])
    obj.run(res=res, x=x, p=-2)


@pytest.mark.api_linalg_cond_parameters
def test_cond6():
    """
    x: n*n； p=fro
    """
    x = randtool("float", -2, 40, [4, 4])
    res = np.array([np.linalg.cond(x, p="fro")])
    obj.run(res=res, x=x, p="fro")


@pytest.mark.api_linalg_cond_parameters
def test_cond7():
    """
    x: n*n； p=nuc
    """
    x = randtool("float", -2, 40, [4, 4])
    res = np.array([np.linalg.cond(x, p="nuc")])
    obj.run(res=res, x=x, p="nuc")


@pytest.mark.api_linalg_cond_parameters
def test_cond8():
    """
    x: n*n； p=1
    """
    x = randtool("float", -2, 40, [4, 4])
    res = np.array([np.linalg.cond(x, p=1)])
    obj.run(res=res, x=x, p=1)


@pytest.mark.api_linalg_cond_parameters
def test_cond9():
    """
    x: n*n； p=-1
    """
    x = randtool("float", -4, 4, [4, 2, 4, 4])
    res = np.array(np.linalg.cond(x, p=-1))
    obj.run(res=res, x=x, p=-1)


@pytest.mark.api_linalg_cond_parameters
def test_cond10():
    """
    x: n*n； p=inf
    """
    x = randtool("float", -4, 4, [4, 2, 4, 4])
    res = np.array(np.linalg.cond(x, p=np.inf))
    obj.run(res=res, x=x, p=np.inf)


@pytest.mark.api_linalg_cond_parameters
def test_cond11():
    """
    x: n*n； p=-inf
    """
    x = randtool("float", -4, 4, [4, 2, 4, 4])
    res = np.array(np.linalg.cond(x, p=-np.inf))
    obj.run(res=res, x=x, p=-np.inf)
