#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test square
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSquare(APIBase):
    """
    test paddle.square api
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.no_grad_var = []
        self.enable_backward = False


obj = TestSquare(paddle.square)


@pytest.mark.api_base_square_vartype
def test_base():
    """
    base
    :return: Tensor
    """
    x = np.array([3, 2])
    res = np.square(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_square_parameters
def test_square1():
    """
    x is Tensor, data type is float32
    """
    x = np.array([3, 2]).astype("float32")
    res = np.square(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_square_parameters
def test_square2():
    """
    x is Tensor, data type is float64
    """
    x = np.array([3, 2]).astype("float64")
    res = np.square(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_square_parameters
def test_square3():
    """
    x is Tensor, data type is float16
    """
    x = np.array([3, 2]).astype("float16")
    res = np.square(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_square_parameters
def test_square4():
    """
    x is Tensor, data type is int32
    """
    x = np.array([3, 2]).astype(np.int32)
    res = np.square(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_square_parameters
def test_square5():
    """
    x is Tensor, data type is int64
    """
    x = np.array([3, 2]).astype(np.int64)
    res = np.square(x)
    obj.run(res=res, x=x)
