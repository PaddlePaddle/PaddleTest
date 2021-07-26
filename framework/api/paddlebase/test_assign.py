#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test assign
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestAssign(APIBase):
    """
    test nonzero
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


obj = TestAssign(paddle.assign)


@pytest.mark.api_base_assign_vartype
def test_assign():
    """
    assign base
    """
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    res = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    obj.base(res=res, x=x)


@pytest.mark.api_base_assign_parameters
def test_assign1():
    """
    x is np.array(bool)
    """
    x = np.array([[True, False, True], [True, False, False], [True, True, True]])
    res = np.array([[True, False, True], [True, False, False], [True, True, True]])
    obj.run(res=res, x=x)


@pytest.mark.api_base_assign_parameters
def test_assign2():
    """
    x is list
    """
    x = [[3.1, 2.1, 2.2], [-1.0, -2.0, -2.2]]
    res = np.array([[3.1, 2.1, 2.2], [-1.0, -2.0, -2.2]])
    obj.run(res=res, x=x)


@pytest.mark.api_base_assign_parameters
def test_assign3():
    """
    x is tuple
    """
    x = ((3.1, 2.1, 2.2), (-1.0, -2.0, -2.2))
    res = np.array([[3.1, 2.1, 2.2], [-1.0, -2.0, -2.2]])
    obj.run(res=res, x=x)


@pytest.mark.api_base_assign_parameters
def test_assign4():
    """
    x is scalar
    """
    x = 42
    res = np.array([42])
    obj.run(res=res, x=x)
