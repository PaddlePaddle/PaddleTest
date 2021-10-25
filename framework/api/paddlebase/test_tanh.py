#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test tanh
"""

from apibase import APIBase
from apibase import randtool

import paddle
import pytest
import numpy as np


class TestTanh(APIBase):
    """
    test tanh
    """

    def hook(self):
        """
        implement
        """
        # np.float16 not support cpu
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestTanh(paddle.tanh)


@pytest.mark.api_base_tanh_vartype
def test_tanh_base():
    """
    base
    """
    x = np.array([2, 3, 4])
    res = np.tanh(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh():
    """
    x=+
    """
    x = randtool("float", 1, 10, [3, 3, 3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh1():
    """
    x=-
    """
    x = randtool("float", -100, -1, [3, 5])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh2():
    """
    x=-/+
    """
    x = randtool("float", -10, 10, [3, 3, 3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh3():
    """
    x = np.array([0])
    """
    x = np.array([0])
    res = np.tanh(x)
    obj.run(res=res, x=x)


# def test_tanh4():
#     """
#     x = np.array([]),[]?None,icafe
#     """
#     x = np.array([])
#     res = np.tanh(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh5():
    """
    x = np.array([1e-9])
    """
    x = np.array([1e-9, 5e3])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh6():
    """
    x=0
    """
    x = np.zeros([2, 3, 4])
    res = np.tanh(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_tanh_parameters
def test_tanh7():
    """
    name="test name"
    """
    name = "test name"
    x = np.zeros([2, 3, 4])
    res = np.tanh(x)
    obj.run(res=res, x=x, name=name)
