#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_fftfreq
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestFFtfreq(APIBase):
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
        # self.delta = 1e-5


obj = TestFFtfreq(paddle.fft.fftfreq)


@pytest.mark.api_fft_fftfreq_vartype
def test_fftfreq_base():
    """
    base
    """
    res = np.array([0.0, 0.50000000, -1.0, -0.50000000])
    obj.base(res=res, n=4, d=0.5)


@pytest.mark.api_fft_fftfreq_parameters
def test_fftfreq0():
    """
    default
    """
    res = np.array([0.0, 1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0, -1.0])
    obj.run(res=res, n=10, d=0.1)
