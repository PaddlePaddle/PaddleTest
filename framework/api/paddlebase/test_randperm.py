#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test randperm
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRandperm(APIBase):
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


obj = TestRandperm(paddle.randperm)


@pytest.mark.api_base_randperm_vartype
def test_randperm_base():
    """
    base
    """
    res = np.array([0, 1, 4, 3, 2, 5])
    n = 6
    obj.base(res=res, n=n)


@pytest.mark.api_base_randperm_parameters
def test_randperm():
    """
    default
    """
    res = np.array([9, 1, 7, 8, 2, 5, 6, 4, 3, 0])
    n = 10
    obj.run(res=res, n=n)


@pytest.mark.api_base_randperm_parameters
def test_randperm1():
    """
    seed = 1
    """
    obj.seed = 1
    res = np.array([6, 3, 7, 8, 9, 2, 1, 5, 4, 0])
    n = 10
    obj.run(res=res, n=n)


@pytest.mark.api_base_randperm_parameters
def test_randperm2():
    """
    dtype = np.float32
    """
    obj.seed = 33
    res = np.array([9.0, 1.0, 7.0, 8.0, 2.0, 5.0, 6.0, 4.0, 3.0, 0.0])
    n = 10
    obj.run(res=res, n=n, dtype=np.float32)


@pytest.mark.api_base_randperm_exception
def test_randperm3():
    """
    exception n < 0 BUG
    """
    obj.seed = 33
    # res = np.array([0.0, 1.0, 6.0, 2.0, 9.0, 3.0, 5.0, 7.0, 4.0, 8.0])
    n = -1
    obj.exception(etype="InvalidArgumentError", n=n, dtype=np.float32)


@pytest.mark.api_base_randperm_exception
def test_randperm4():
    """
    exception dtype = np.int8 BUG
    """
    obj.seed = 33
    # res = np.array([0.0, 1.0, 6.0, 2.0, 9.0, 3.0, 5.0, 7.0, 4.0, 8.0])
    n = -1
    obj.exception(etype="NotFoundError", n=n, dtype=np.int8)
