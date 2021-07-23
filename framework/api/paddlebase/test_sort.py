#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sort
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSort(APIBase):
    """
    test sort
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = True


obj = TestSort(paddle.sort)


@pytest.mark.api_base_sort_vartype
def test_sort_base():
    """
    sort_base
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.sort(x_data, axis=0)
    obj.base(res=res, x=x_data, axis=0)


@pytest.mark.api_base_sort_parameters
def test_sort_axis0():
    """
    axis=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.sort(x_data, axis=0)
    obj.run(res=res, x=x_data, axis=0)


@pytest.mark.api_base_sort_parameters
def test_sort_axis1():
    """
    axis=1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.sort(x_data, axis=1)
    obj.run(res=res, x=x_data, axis=1)


@pytest.mark.api_base_sort_parameters
def test_sort_axis2():
    """
    axis=-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.sort(x_data, axis=-1)
    obj.run(res=res, x=x_data, axis=-1)


@pytest.mark.api_base_sort_parameters
def test_argsort_descending():
    """
    descending=True
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.array([[[8.0, 9.0], [10.0, 11.0]], [[4.0, 5.0], [6.0, 7.0]], [[0.0, 1.0], [2.0, 3.0]]])
    obj.run(res=res, x=x_data, axis=0, descending=True)
