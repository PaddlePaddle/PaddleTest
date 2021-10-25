#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test prod
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestProd(APIBase):
    """
    test prod
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = True


obj = TestProd(paddle.prod)


@pytest.mark.api_base_prod_vartype
def test_prod_base():
    """
    base
    axis=None
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = [np.prod(x)]
    obj.base(res=res, x=x)


@pytest.mark.api_base_prod_parameters
def test_prod1():
    """
    axis<rank(x)
    """
    x = np.array([[-0.8, -0.4], [0.7, 0.9]])
    axis = 1
    res = np.prod(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_prod_parameters
def test_prod2():
    """
    axis>rank(x)
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = 3
    obj.exception(mode="c", etype="InvalidArgumentError", x=x, axis=axis)


@pytest.mark.api_base_prod_parameters
def test_prod3():
    """
    axis=negtive num
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = -1
    res = np.prod(x, axis=axis)
    obj.run(res=res, x=x, axis=axis)


@pytest.mark.api_base_prod_parameters
def test_prod4():
    """
    axis=list
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = [0, 1]
    res = [0.2016]
    obj.run(res=res, x=x, axis=axis)


def test_prod5():
    """
    axis=tuple
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    axis = (0, 1)
    res = [0.2016]
    obj.run(res=res, x=x, axis=axis)


def test_prod6():
    """
    dtype=float32
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = [np.prod(x)]
    obj.base(res=res, x=x, dtype="float32")


def test_prod7():
    """
    keepdim=True
    """
    x = np.array([[0.8, 0.4], [0.7, 0.9]])
    res = np.array([[0.56, 0.36]])
    obj.base(res=res, x=x, axis=0, keepdim=True)


class TestProd1(APIBase):
    """
    test prod
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64]
        self.enable_backward = False


obj1 = TestProd1(paddle.prod)


def test_prod8():
    """
    dtype=int64
    """
    x = np.array([[8, 4], [7, 9]])
    res = [np.prod(x)]
    obj1.base(res=res, x=x, dtype="int64")


def test_prod9():
    """
    input is int32, int64
    """
    x = np.array([[3, 5], [6, 2]])
    res = [np.prod(x)]
    obj1.run(res=res, x=x)
