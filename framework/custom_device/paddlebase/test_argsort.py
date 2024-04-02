#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test argsort
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestArgsort(APIBase):
    """
    test argsort
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = False
        # enable check grad
        self.enable_backward = False


obj = TestArgsort(paddle.argsort)


@pytest.mark.api_base_argsort_vartype
def test_argsort_base():
    """
    argsort_base
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.argsort(x_data, axis=0)
    obj.base(res=res, x=x_data, axis=0)


@pytest.mark.api_base_argsort_parameters
def test_argsort_axis0():
    """
    axis=0
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.argsort(x_data, axis=0)
    obj.run(res=res, x=x_data, axis=0)


@pytest.mark.api_base_argsort_parameters
def test_argsort_axis1():
    """
    axis=1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.argsort(x_data, axis=1)
    obj.run(res=res, x=x_data, axis=1)


@pytest.mark.api_base_argsort_parameters
def test_argsort_axis2():
    """
    axis=-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.argsort(x_data, axis=-1)
    obj.run(res=res, x=x_data, axis=-1)


@pytest.mark.api_base_argsort_parameters
def test_argsort_descending():
    """
    axis=-1
    """
    x_data = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = np.array([[[2, 2], [2, 2]], [[1, 1], [1, 1]], [[0, 0], [0, 0]]])
    obj.run(res=res, x=x_data, axis=0, descending=True)
