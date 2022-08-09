#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_any
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestAny(APIBase):
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


obj = TestAny(paddle.any)


@pytest.mark.api_base_any_parameters
def test_any0():
    """
    default
    """
    x = np.random.randint(-4, 4, (10,))
    res = np.array([np.any(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_any_parameters
def test_any1():
    """
    x: 2d-tensor
    """
    x = np.random.randint(-4, 4, (10, 10))
    res = np.array([np.any(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_any_parameters
def test_any2():
    """
    x: 3d-tensor
    """
    x = np.random.randint(-4, 4, (3, 4, 2))
    res = np.array([np.any(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_any_parameters
def test_any3():
    """
    x: 4d-tensor
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.array([np.any(x)])
    obj.base(res=res, x=x)


@pytest.mark.api_base_any_parameters
def test_any4():
    """
    x: 4d-tensor
    axis: 1
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.any(x, axis=1)
    obj.base(res=res, x=x, axis=1)


@pytest.mark.api_base_any_parameters
def test_any5():
    """
    x: 4d-tensor
    axis: -1
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.any(x, axis=-1)
    obj.base(res=res, x=x, axis=-1)


@pytest.mark.api_base_any_parameters
def test_any5():
    """
    x: 4d-tensor
    axis: (0, 1)
    """
    x = np.random.randint(-4, 4, (2, 4, 4, 2))
    res = np.any(x, axis=(0, 1))
    obj.base(res=res, x=x, axis=(0, 1))


@pytest.mark.api_base_any_parameters
def test_any6():
    """
    x: 2d-tensor
    keepdim = True
    """
    x = np.random.randint(-4, 4, (2, 4))
    res = np.any(x, keepdims=True)
    obj.base(res=res, x=x, keepdim=True)
