#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
rfftfreq
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRfftfreq(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float64, np.float32]
        # self.delta = 1e-3
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False


obj = TestRfftfreq(paddle.fft.rfftfreq)


@pytest.mark.api_fft_rfftn_vartype
def test_rfftfreq_base():
    """
    base
    :return:
    """
    x = np.array([3, 1, 2, 2, 3], dtype=float)
    n = x.size
    res = [0.0, 0.66666669, 1.33333337]
    obj.base(res=res, n=n, d=0.3)


@pytest.mark.api_fft_rfftn_parameters
def test_rfftfreq():
    """
    base
    :return:
    """
    x = np.array([3, 1, 2, 2, 3], dtype=float)
    n = x.size
    res = [0.0, 0.66666669, 1.33333337]
    obj.run(res=res, n=n, d=0.3)


@pytest.mark.api_fft_rfftn_parameters
def test_rfftfreq1():
    """
    n = 8
    :return:
    """
    x = np.array([3, 1, 2, 2, 3, 2, 1, 3], dtype=float)
    n = x.size
    res = [0.0, 0.41666666, 0.8333333, 1.25, 1.6666666]
    obj.run(res=res, n=n, d=0.3)


@pytest.mark.api_fft_rfftn_parameters
def test_rfftfreq2():
    """
    n = 8
    :return:
    """
    x = np.array([3, 1, 2, 2, 3, 2, 1, 3], dtype=float)
    n = x.size
    res = [0, 0.125, 0.25, 0.375, 0.5]
    obj.run(res=res, n=n, d=1)
