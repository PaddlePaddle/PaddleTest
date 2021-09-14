#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_gather.py
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestGather(APIBase):
    """
    test gather
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        self.enable_backward = False
        self.no_grad_var = ["index", "axis"]


obj = TestGather(paddle.gather)


@pytest.mark.api_base_gather_vartype
def test_gather_base():
    """
    base
    index is int32
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2, 1, 0]).astype("int32")
    res = np.array([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
    obj.base(res=res, x=x, index=index, axis=0)


@pytest.mark.api_base_gather_vartype
def test_gather1():
    """
    index is int64
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2, 1, 0]).astype("int64")
    res = np.array([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
    obj.base(res=res, x=x, index=index, axis=0)


@pytest.mark.api_base_gather_parameters
def test_gather2():
    """
    axis = 1
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2, 1, 0]).astype("int64")
    res = np.array([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6]])
    obj.base(res=res, x=x, index=index, axis=1)


@pytest.mark.api_base_gather_parameters
def test_gather3():
    """
    axis is tensor
    """
    x = np.arange(9).reshape(3, 3)
    index = np.array([0, 2, 1, 0]).astype("int64")
    axis = np.array([1]).astype("int64")
    res = np.array([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6]])
    obj.base(res=res, x=x, index=index, axis=axis)
