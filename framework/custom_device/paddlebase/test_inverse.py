#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test inverse
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestInverse(APIBase):
    """
    test inverse
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj = TestInverse(paddle.inverse)


@pytest.mark.api_base_inverse_vartype
def test_inverse_base():
    """
    inverse_base
    """
    x_data = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[4.0, 0, 0], [0, 4, 0], [0, 0, 4]]])
    res = np.array([[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]])
    obj.base(res=res, x=x_data)


@pytest.mark.api_base_inverse_parameters
def test_inverse_diagonal():
    """
    diagonal matrix inverse
    """
    # x_data = np.arange(18).reshape((2, 3, 3)).astype(np.float32)
    x_data = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[4.0, 0, 0], [0, 4, 0], [0, 0, 4]]])
    res = np.array([[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]]])
    obj.run(res=res, x=x_data)
