#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test trace
"""

from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestTrace(APIBase):
    """
    test trace
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16, np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = True


obj2 = TestTrace(paddle.trace)


@pytest.mark.api_base_trace_parameters
def test_trace_offset1():
    """
    offset=0, axis1=0, axis2=1
    """
    x_data = np.arange(12).reshape((3, 4)).astype(np.float32)
    res = np.array([15.0])
    obj2.run(res=res, x=x_data, offset=0, axis1=0, axis2=1)


@pytest.mark.api_base_trace_parameters
def test_trace_offset2():
    """
    offset=-1, axis1=0, axis2=1
    """
    x_data = np.arange(12).reshape((3, 4)).astype(np.float32)
    res = np.array([13.0])
    obj2.run(res=res, x=x_data, offset=-1, axis1=0, axis2=1)


@pytest.mark.api_base_trace_parameters
def test_trace_offset3():
    """
    offset=1, axis1=0, axis2=1
    """
    x_data = np.arange(12).reshape((3, 4)).astype(np.float32)
    res = np.array([18.0])
    obj2.run(res=res, x=x_data, offset=1, axis1=0, axis2=1)


@pytest.mark.api_base_trace_parameters
def test_trace_axis1():
    """
    offset=0, axis1=-3, axis2=-2
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([8.0, 10.0])
    obj2.run(res=res, x=x_data, offset=0, axis1=-3, axis2=-2)


@pytest.mark.api_base_trace_parameters
def test_trace_axis2():
    """
    offset=-1, axis1=2, axis2=-2
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([7.0, 19.0])
    obj2.run(res=res, x=x_data, offset=-1, axis1=2, axis2=-2)


@pytest.mark.api_base_trace_parameters
def test_trace_axis3():
    """
    offset=1, axis1=0, axis2=2
    """
    x_data = np.arange(12).reshape((2, 3, 2)).astype(np.float32)
    res = np.array([1.0, 3.0, 5])
    obj2.run(res=res, x=x_data, offset=1, axis1=0, axis2=2)
