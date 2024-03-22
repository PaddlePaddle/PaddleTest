#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.triu
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestTriu(APIBase):
    """
    test triu
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


obj = TestTriu(paddle.triu)


@pytest.mark.api_base_triu_vartype
def test_triu_base():
    """
    diagonal=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data)
    obj.base(res=res, x=x_data, diagonal=0)


@pytest.mark.api_base_triu_parameters
def test_triu_diagonal1():
    """
    diagonal=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data, k=0)
    obj.run(res=res, x=x_data, diagonal=0)


@pytest.mark.api_base_triu_parameters
def test_triu_diagonal2():
    """
    diagonal=1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data, k=1)
    obj.run(res=res, x=x_data, diagonal=1)


@pytest.mark.api_base_triu_parameters
def test_triu_diagonal3():
    """
    diagonal=-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data, k=-1)
    obj.run(res=res, x=x_data, diagonal=-1)


@pytest.mark.api_base_triu_parameters
def test_triu_diagonal4():
    """
    diagonal<-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data, k=-5)
    obj.run(res=res, x=x_data, diagonal=-5)


@pytest.mark.api_base_triu_parameters
def test_triu_diagonal5():
    """
    diagonal>-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.triu(x_data, k=5)
    obj.run(res=res, x=x_data, diagonal=5)
