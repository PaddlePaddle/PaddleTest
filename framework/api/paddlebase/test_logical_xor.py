#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expxortab:ft=python
"""
test logical xor
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogicalXor(APIBase):
    """
    test logical_xor
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


obj = TestLogicalXor(paddle.logical_xor)


@pytest.mark.api_base_logical_xor_vartype
def test_logical_xor_1D_tensor():
    """
    logical_xor_1D_tensor
    """
    x_data = np.array([True])
    y_data = np.array([True, False, True, False])
    res = np.logical_xor(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_logical_xor_parameters
def test_logical_xor_broadcast_1():
    """
    logical_xor_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    y_data = np.arange(0, 6).reshape((1, 2, 3)).astype(np.bool)
    res = np.logical_xor(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)


@pytest.mark.api_base_logical_xor_parameters
def test_logical_xor_broadcast_2():
    """
    logical_xor_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    y_data = np.arange(0, 4).reshape((2, 2)).astype(np.bool)
    res = np.logical_xor(x_data, y_data)
    obj.run(res=res, x=x_data, y=y_data)
