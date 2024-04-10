#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test tanh_
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestTanh(APIBase):
    """
    test tanh_
    """

    def hook(self):
        """
        implement
        """
        # np.float16 not support cpu
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestTanh(paddle.tanh_)


@pytest.mark.api_base_tanh__vartype
def test_tanh__base():
    """
    base
    """
    x = np.array([2, 3, 4])
    res = np.tanh(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_():
    """
    x=+
    """
    x = randtool("float", 1, 10, [3, 3, 3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_1():
    """
    x=-
    """
    x = randtool("float", -100, -1, [3, 5])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_2():
    """
    x=-/+
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_3():
    """
    x = np.array([0])
    """
    x = np.array([0])
    res = np.tanh(x)
    obj.run(res=res, x=x)


# def test_tanh_4():
#     """
#     x = np.array([]),[]?None,icafe
#     """
#     x = np.array([])
#     res = np.tanh(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_5():
    """
    x = np.array([1e-9])
    """
    x = np.array([1e-9, 5e3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_6():
    """
    x=0
    """
    x = np.zeros([2, 3, 4])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh__parameters
def test_tanh_7():
    """
    name="test name"
    """
    name = "test name"
    x = np.zeros([2, 3, 4])
    res = np.tanh(x)
    obj.run(res=res, x=x, name=name)
