#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_unique
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestUnique(APIBase):
    """
    test unique
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False


obj = TestUnique(paddle.unique)


@pytest.mark.api_base_unique_vartype
def test_unique_base():
    """
    base
    use_default_value
    """
    x = np.array([2.3, 3.9, 3.2, 1.3, 5.4, 3.8])
    res = np.unique(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_unique_parameters
def test_unique1():
    """
    base
    use_axis
    """
    x = np.array([[0.4, 0.4], [0.7, 0.9]])
    res = np.unique(x, axis=1)
    obj.base(res=res, x=x, axis=1)


class TestUnique1(APIBase):
    """
    test unique
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        self.enable_backward = False
        self.debug = True


obj1 = TestUnique1(paddle.unique)


@pytest.mark.api_base_unique_vartype
def test_unique2():
    """
    input is int32,int64
    axis=0
    """
    x = np.array([[8, 4], [7, 9]])
    res = np.unique(x, axis=0)
    obj1.base(res=res, x=x, axis=0)


@pytest.mark.api_base_unique_exception
def test_unique3():
    """
    invalid axis
    """
    x = np.array([[8, 4], [7, 9]])
    obj1.exception(mode="c", etype="InvalidArgument", x=x, axis=3)
