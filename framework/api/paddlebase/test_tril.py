#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jiaxiao01
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestTril(APIBase):
    """
    test tril
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestTril(paddle.tril)


@pytest.mark.api_base_tril_vartype
def test_tril_base():
    """
    diagonal=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data)
    obj.base(res=res, x=x_data, diagonal=0)


@pytest.mark.api_base_tril_parameters
def test_tril_diagonal1():
    """
    diagonal=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data, k=0)
    obj.run(res=res, x=x_data, diagonal=0)


@pytest.mark.api_base_tril_parameters
def test_tril_diagonal2():
    """
    diagonal=1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data, k=1)
    obj.run(res=res, x=x_data, diagonal=1)


@pytest.mark.api_base_tril_parameters
def test_tril_diagonal3():
    """
    diagonal=-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data, k=-1)
    obj.run(res=res, x=x_data, diagonal=-1)


@pytest.mark.api_base_tril_parameters
def test_tril_diagonal4():
    """
    diagonal<-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data, k=-5)
    obj.run(res=res, x=x_data, diagonal=-5)


@pytest.mark.api_base_tril_parameters
def test_tril_diagonal5():
    """
    diagonal>-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.tril(x_data, k=5)
    obj.run(res=res, x=x_data, diagonal=5)
