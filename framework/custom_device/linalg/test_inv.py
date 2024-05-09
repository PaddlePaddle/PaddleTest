#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_inv
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestInv(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-4


obj = TestInv(paddle.linalg.inv)


@pytest.mark.api_linalg_inv_vartype
def test_inv_base():
    """
    base
    """
    x = np.random.rand(4, 4)
    res = np.linalg.inv(x)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_inv_parameters
def test_inv0():
    """
    x: 3-d tensor
    """
    obj.enable_backward = False
    x = np.random.rand(2, 2, 2)
    res = np.linalg.inv(x)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_det_parameters
def test_det1():
    """
    X: 4-d tensor
    """
    x = np.random.rand(5, 3, 4, 4)
    res = np.linalg.inv(x)
    obj.run(res=res, x=x)
