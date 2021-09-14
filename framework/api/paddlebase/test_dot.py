#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
test paddle.dot
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestDot(APIBase):
    """
    test dot
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestDot(paddle.dot)


@pytest.mark.api_base_dot_vartype
def test_dot_base():
    """
    dot_base
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    y_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    res = np.array([91.0])
    obj.base(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_dot_parameters
def test_dot_1D_tensor():
    """
    dot_1D_tensor
    """
    x_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    y_data = np.arange(1, 7).reshape((6,)).astype(np.float32)
    res = np.array([91.0])
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_dot_parameters
def test_dot_2D_tensor():
    """
    dot_2D_tensor
    """
    x_data = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    y_data = np.arange(1, 7).reshape((2, 3)).astype(np.float32)
    res = np.array([[14.0], [77.0]])
    obj.run(res=res, x=x_data, y=y_data)
