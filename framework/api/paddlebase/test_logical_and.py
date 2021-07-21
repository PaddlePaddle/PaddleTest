#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test logical and
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogicalAnd(APIBase):
    """
    test logical_and
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


obj = TestLogicalAnd(paddle.logical_and)


@pytest.mark.api_base_logical_and_vartype
def test_logical_and_1D_tensor():
    """
    logical_and_1D_tensor
    """
    x_data = np.array([True])
    y_data = np.array([True, False, True, False])
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_logical_and_parameters
def test_logical_and_broadcast_1():
    """
    logical_and_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.bool)
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_logical_and_parameters
def test_logical_and_broadcast_2():
    """
    logical_and_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.bool)
    res = np.logical_and(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
