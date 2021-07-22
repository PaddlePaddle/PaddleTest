#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test equal
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestEqual(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestEqual(paddle.equal)


@pytest.mark.api_base_equal_vartype
def test_equal_base():
    """
    base
    """
    x = randtool("int", -10, 10, [3, 3, 3])
    y = x
    res = np.equal(x, y)
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal():
    """
    default
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = x
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal1():
    """
    test broadcast
    """
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([1, 2, 3])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal2():
    """
    test broadcast1
    """
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 2, 3]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal3():
    """
    test broadcast2
    """
    x = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    y = np.array([[1, 2, 3]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal4():
    """
    test broadcast3  x.shape < y.shape
    """
    x = np.array([[1, 2, 3]])
    y = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal5():
    """
    x != y
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 3])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal6():
    """
    x != y broadcast
    """
    x = randtool("float", -10, 10, [3, 3, 3, 1])
    y = randtool("float", -10, 10, [3, 3, 1])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal7():
    """
    x != y broadcast  x.shape < y.shape
    """
    x = randtool("float", -10, 10, [3, 3, 1])
    y = randtool("float", -10, 10, [3, 3, 3, 1])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_exception
def test_equal8():
    """
    exception broadcast error
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("float", -10, 10, [3, 3, 2])
    obj.exception(etype="InvalidArgumentError", x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal9():
    """
    test broadcast1
    x != y
    x.type = bool
    """
    x = np.array([[True, False, True], [True, False, True], [True, False, True]])
    y = np.array([True, False, True])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal10():
    """
    test broadcast2
    x.type = bool
    """
    x = np.array([[True, False, True], [True, False, True], [True, False, True]])
    y = np.array([[True, False, True]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal11():
    """
    test broadcast3  y.shape > x.shape
    x.type = bool
    """
    x = np.array([[True, False, True]])
    y = np.array([[[[[True, False, True], [True, False, True], [True, False, True]]]]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal12():
    """
    test broadcast2
    x != y
    x.type = bool
    """
    x = np.array([[True, False, True], [True, False, True], [True, False, True]])
    y = np.array([[True, False, True], [False, False, False], [True, True, False]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal13():
    """
    test broadcast2
    x.shape > y.shape
    x.type = bool
    """
    x = np.array([[[[[[True, False, True], [True, False, True], [True, False, True]]]]]])
    y = np.array([[True, False, True], [True, False, True], [True, False, True]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal14():
    """
    test broadcast2
    x.shape > y.shape
    x != y
    x.type = bool
    """
    x = np.array([[[[[[True, False, True], [True, False, True], [True, False, True]]]]]])
    y = np.array([[True, False, True], [True, False, True], [True, False, True]])
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_parameters
def test_equal15():
    """
    x = y
    x.type = bool
    """
    x = np.array([[[[[[True, False, True], [True, False, True], [True, False, True]]]]]])
    y = x
    res = np.equal(x, y)
    obj.run(res=res, x=x, y=y)
