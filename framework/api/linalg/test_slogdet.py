#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_slogdet
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
        self.types = [np.float64, np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1


obj = TestDet(paddle.linalg.slogdet)


@pytest.mark.api_linalg_slogdet_vartype
def test_slogdet_base():
    """
    base
    """
    x = np.random.rand(14, 14) * 100
    res = np.array(np.linalg.slogdet(x)).reshape(2, 1)
    obj.base(res=res, x=x)


@pytest.mark.api_linalg_slogdet_parameters
def test_slogdet0():
    """
    default
    """
    x = np.random.rand(4, 4)
    res = np.array(np.linalg.slogdet(x)).reshape(2, 1)
    obj.run(res=res, x=x)


@pytest.mark.api_linalg_slogdet_parameters
def test_slogdet1():
    """
    multi_dim
    """
    x = np.random.rand(3, 4, 4)
    res = np.array(np.linalg.slogdet(x))
    obj.run(res=res, x=x)
