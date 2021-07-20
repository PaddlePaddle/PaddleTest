#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test equal all
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestEqualAll(APIBase):
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


obj = TestEqualAll(paddle.equal_all)


@pytest.mark.api_base_equal_all_vartype
def test_equal_all_base():
    """
    base
    """
    x = randtool("int", -10, 10, [3, 3, 3])
    y = x
    res = np.array([True])
    obj.base(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all():
    """
    x = float, y = x
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = x
    res = np.array([True])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all1():
    """
    x = float, y != x
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("int", -10, 10, [3, 3, 3])
    res = np.array([False])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all2():
    """
    x = float, y != x and y.shape != x.shape
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    y = randtool("int", -10, 10, [3, 3])
    res = np.array([False])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all4():
    """
    x = float, y broadcast = x but y.shape != x.shape
    """
    x = np.array([[3, 3, 3], [3, 3, 3]])
    y = np.array([[3, 3, 3]])
    res = np.array([False])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all5():
    """
    x = float, y broadcast = x but y.shape != x.shape
    """
    x = np.array([[3, 3, 3], [3, 3, 3]])
    y = np.array([["S", 3, 3]])
    obj.exception(etype=ValueError, mode="python", x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all6():
    """
    x = bool, y broadcast = x but y.shape != x.shape
    """
    x = np.array([[True, False, True], [True, False, True]])
    y = np.array([[True, False, True]])
    res = np.array([False])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all7():
    """
    x = bool, y = x but y.shape != x.shape
    """
    x = np.array([[[True, False, True], [True, False, True]]])
    y = np.array([[True, False, True], [True, False, True]])
    res = np.array([False])
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_equal_all_parameters
def test_equal_all8():
    """
    x = bool, y = x
    """
    x = np.array([[[[[True, False, True], [True, False, True]]]]])
    y = x
    res = np.array([True])
    obj.run(res=res, x=x, y=y)
