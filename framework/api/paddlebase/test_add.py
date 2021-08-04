#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test abs
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAdd(APIBase):
    """
    test abs
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestAdd(paddle.add)
obj1 = TestAdd(paddle.add)
obj1.types = [np.int32, np.int64]


@pytest.mark.api_base_abs_vartype
def test_add_base():
    """
    base
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    y = randtool("float", -10, 10, (3, 3, 3))
    res = np.add(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_abs_vartype
def test_add_base1():
    """
    base
    """
    x = randtool("int", -10, 10, (3, 3, 3))
    y = randtool("int", -10, 10, (3, 3, 3))
    res = np.add(x, y)
    obj1.base(res=res, x=x, y=y)


@pytest.mark.api_base_abs_parameters
def test_add():
    """
    x is int, y is float
    """
    x = randtool("int", -10, 10, (3, 3, 3))
    y = randtool("float", -10, 10, (3, 3, 3))
    res = np.add(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_abs_parameters
def test_add1():
    """
    x is int, y is int
    """
    x = randtool("int", -10, 10, (3, 3, 3))
    y = randtool("int", -10, 10, (3, 3, 3))
    res = np.add(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_abs_parameters
def test_add2():
    """
    x is float, y is float, broadcast
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    y = randtool("float", -10, 10, (3, 3, 1))
    res = np.add(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_abs_parameters
def test_add3():
    """
    x is float, y is float, broadcast
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    y = randtool("float", -10, 10, (3, 1, 3))
    res = np.add(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_abs_parameters
def test_add4():
    """
    x is float, y is float, broadcast
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    y = randtool("float", -10, 10, (1, 3, 3))
    res = np.add(x, y)
    obj.run(res=res, x=x, y=y)
