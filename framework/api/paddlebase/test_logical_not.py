#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expnottab:ft=python
"""
test logical not
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogicalNot(APIBase):
    """
    test logical_not
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


obj = TestLogicalNot(paddle.logical_not)


@pytest.mark.api_base_logical_not_vartype
def test_logical_not_1D_tensor():
    """
    logical_not_1D_tensor
    """
    x_data = np.array([True])
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_logical_not_parameters
def test_logical_not_broadcast_1():
    """
    logical_not_broadcast_1
    """
    x_data = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.bool)
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)


@pytest.mark.api_base_logical_not_parameters
def test_logical_not_broadcast_2():
    """
    logical_not_broadcast_2
    """
    x_data = np.arange(1, 3).reshape((1, 2)).astype(np.bool)
    res = np.logical_not(x_data)
    obj.run(res=res, x=x_data)
