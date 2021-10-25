#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test logsumexp
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLogsumexp(APIBase):
    """
    test logsumexp
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
        self.delta = 5e-3


obj = TestLogsumexp(paddle.logsumexp)


@pytest.mark.api_base_logsumexp_vartype
def test_logsumexp_base():
    """
    logsumexp_base
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([[1.3132616, 3.3132617, 5.3132615], [7.313262, 9.313262, 11.313262]])
    obj.base(res=res, x=x_data, axis=2)


@pytest.mark.api_base_logsumexp_parameters
def test_logsumexp_axis1():
    """
    axis=2, keepdim=False
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([[1.3132616, 3.3132617, 5.3132615], [7.313262, 9.313262, 11.313262]])
    obj.run(res=res, x=x_data, axis=2)


@pytest.mark.api_base_logsumexp_parameters
def test_logsumexp_axis2():
    """
    axis=[0, 1], keepdim=False
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([10.145408, 11.145408])
    obj.run(res=res, x=x_data, axis=[0, 1])


@pytest.mark.api_base_logsumexp_parameters
def test_logsumexp_keepdim1():
    """
    axis=0, keepdim=True
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([[[6.0024757, 7.0024757], [8.002476, 9.002476], [10.002476, 11.002476]]])
    obj.run(res=res, x=x_data, axis=0, keepdim=True)


@pytest.mark.api_base_logsumexp_parameters
def test_logsumexp_keepdim2():
    """
    axis=0, keepdim=True
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([[[10.145408, 11.145408]]])
    obj.run(res=res, x=x_data, axis=[0, 1], keepdim=True)
