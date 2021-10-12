#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_det
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestDet(APIBase):
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
        self.enable_backward = False
        self.delta = 1e-3


obj = TestDet(paddle.linalg.det)


@pytest.mark.api_linalg_det_vartype
def test_det_base():
    """
    base
    """
    x = np.random.rand(4, 4) * 10
    res = np.linalg.det(x).reshape(1)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_det_parameters
def test_det0():
    """
    default
    """
    x = np.random.rand(4, 4)
    res = np.linalg.det(x).reshape(1)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_det_parameters
def test_det1():
    """
    multi_dim
    """
    x = np.random.rand(3, 4, 4)
    res = np.linalg.det(x)
    obj.run(res=res, x=x)
