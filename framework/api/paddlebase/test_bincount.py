#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_bincount
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestBincount(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestBincount(paddle.bincount)


@pytest.mark.api_base_bincount_vartype
def test_bincount_base():
    """
    base
    """
    x = randtool("int", 0, 4, (4,))
    res = np.bincount(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_bincount_parameters
def test_bincount0():
    """
    default
    """
    x = randtool("int", 0, 5, (10,))
    res = np.bincount(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_bincount_parameters
def test_bincount1():
    """
    weight : int
    """
    x = randtool("int", 0, 10, (10,))
    weight = randtool("int", -4, 4, (10,))
    res = np.bincount(x, weight)
    obj.run(res=res, x=x, weights=weight)


@pytest.mark.api_base_bincount_parameters
def test_bincount2():
    """
    weight : float
    """
    paddle.disable_static()
    x = randtool("int", 0, 4, (4,))
    weight = randtool("float", 2, 4, (4,))
    res = np.bincount(x, weights=weight)
    x = paddle.to_tensor(x)
    weight = paddle.to_tensor(weight)
    api_res = paddle.bincount(x, weights=weight)
    assert np.allclose(res, api_res.numpy())


@pytest.mark.api_base_bincount_parameters
def test_bincount3():
    """
    minlength = 4
    """
    x = randtool("int", 0, 10, (10,))
    minlength = 4
    res = np.bincount(x, minlength=minlength)
    obj.run(res=res, x=x, minlength=minlength)
