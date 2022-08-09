#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_all
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestAll(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.bool]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


obj = TestAll(paddle.all)


@pytest.mark.api_base_all_parameters
def test_all0():
    """
    default
    """
    x = np.random.randint(-4, 4, (10,))
    res = np.array([np.all(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_all_parameters
def test_all1():
    """
    x: 2d-tensor
    """
    x = np.random.randint(-4, 4, (10, 10))
    res = np.array([np.all(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_all_parameters
def test_all2():
    """
    x: 3d-tensor
    """
    x = np.random.randint(-4, 4, (3, 4, 2))
    res = np.array([np.all(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_all_parameters
def test_all3():
    """
    x: 4d-tensor
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.array([np.all(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_all_parameters
def test_all4():
    """
    x: 4d-tensor
    axis: 1
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.all(x, axis=1)
    obj.base(res=res, x=x, axis=1)


@pytest.mark.api_base_all_parameters
def test_all5():
    """
    x: 4d-tensor
    axis: -1
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.all(x, axis=-1)
    obj.base(res=res, x=x, axis=-1)


@pytest.mark.api_base_all_parameters
def test_all5():
    """
    x: 4d-tensor
    axis: (0, 1)
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.all(x, axis=(0, 1))
    obj.base(res=res, x=x, axis=(0, 1))


@pytest.mark.api_base_all_parameters
def test_all6():
    """
    x: 2d-tensor
    keepdim = True
    """
    x = np.random.randint(-4, 4, (2, 4))
    res = np.all(x, keepdims=True)
    obj.base(res=res, x=x, keepdim=True)
