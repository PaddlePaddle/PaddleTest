#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test multiply
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class Multiply(APIBase):
    """
    test multiply
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = Multiply(paddle.multiply)


@pytest.mark.api_base_multiply_vartype
def test_multiply_base():
    """
    multiply_base
    """
    x_data = np.array([2.0])
    y_data = np.array([1, -1, 4, 5])
    res = np.multiply(x_data, y_data)
    obj.base(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_multiply_parameters
def test_multiply_1D_tensor():
    """
    multiply_1D_tensor
    """
    x_data = np.array([2.0])
    y_data = np.array([1, -1, 4, 5])
    res = np.multiply(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_multiply_parameters
def test_multiply_broadcast_1():
    """
    multiply_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.float32)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.float32)
    res = np.multiply(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_multiply_parameters
def test_multiply_broadcast_2():
    """
    multiply_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.float32)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.float32)
    res = np.multiply(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
