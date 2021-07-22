#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test randint
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestRandint(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # self.places = [fluid.CPUPlace()]
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestRandint(paddle.randint)


@pytest.mark.api_base_randint_vartype
def test_randint_base():
    """
    base
    """
    res = np.array([[9, 9], [5, 8]])
    obj.base(res=res, low=0, high=10, shape=[2, 2])


@pytest.mark.api_base_randint_parameters
def test_randint():
    """
    default
    """
    res = np.array([[[9, 9], [5, 8], [3, 4]], [[9, 4], [4, 7], [2, 9]], [[8, 4], [6, 2], [9, 3]]])
    obj.run(res=res, low=0, high=10, shape=[3, 3, 2])


@pytest.mark.api_base_randint_parameters
def test_randint1():
    """
    seed = 1
    """
    obj.seed = 1
    res = np.array([[0, 7], [6, 0]])
    obj.run(res=res, low=0, high=10, shape=[2, 2])


@pytest.mark.api_base_randint_exception
def test_randint2():
    """
    exception shape
    """
    # res = np.array([[0, 7], [6, 0]])
    obj.exception(etype=AttributeError, mode="python", low=0, high=10, shape="2")


@pytest.mark.api_base_randint_parameters
def test_randint3():
    """
    dtype is int64
    """
    obj.seed = 1
    res = np.array([[0, 7], [6, 0]])
    obj.run(res=res, low=0, high=10, shape=[2, 2], dtype=np.int64)


@pytest.mark.api_base_randint_exception
def test_randint4():
    """
    exception dtype is float BUG
    """
    # res = np.array([[9, 9], [5, 8]])
    obj.exception(etype="NotFoundError", low=0, high=10, shape=[2, 2], dtype=np.float32)


@pytest.mark.api_base_randint_exception
def test_randint5():
    """
    exception low > high
    """
    # res = np.array([[9, 9], [5, 8]])
    obj.exception(etype="InvalidArgumentError", low=100, high=10, shape=[2, 2])


@pytest.mark.api_base_randint_exception
def test_randint6():
    """
    exception low > high
    """
    # res = np.array([[9, 9], [5, 8]])
    obj.exception(etype=ValueError, mode="python", low=0, shape=[2, 2])
