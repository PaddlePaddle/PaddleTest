#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test arange
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestArange(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32, np.int64, np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        self.no_grad_var = ("start", "end", "step")


obj = TestArange(paddle.arange)


@pytest.mark.api_base_arange_vartype
def test_arange_base():
    """
    base
    """
    start = 5
    res = np.arange(start)
    obj.base(res=res, start=start)


@pytest.mark.api_base_arange_parameters
def test_arange():
    """
    default
    """
    start = 5
    end = 10
    res = np.arange(start, end)
    obj.run(res=res, start=start, end=end)


@pytest.mark.api_base_arange_parameters
def test_arange1():
    """
    start=1, end=20, step=3
    """
    start = 1
    end = 20
    step = 3
    res = np.arange(start, end, step)
    obj.run(res=res, start=start, end=end, step=step)


@pytest.mark.api_base_arange_parameters
def test_arange2():
    """
    start=1, end=20, step=3, dtype=np.float32
    """
    start = 1
    end = 20
    step = 3
    dtype = np.float32
    res = np.arange(start, end, step).astype(dtype)
    obj.run(res=res, start=start, end=end, step=step, dtype=dtype)


@pytest.mark.api_base_arange_parameters
def test_arange3():
    """
    start=1, end=20, step=3, dtype=np.int64
    """
    start = 1
    end = 20
    step = 3
    dtype = np.int64
    res = np.arange(start, end, step).astype(dtype)
    obj.run(res=res, start=start, end=end, step=step, dtype=dtype)


@pytest.mark.api_base_arange_parameters
def test_arange4():
    """
    (np)start=1, end=20, step=3, dtype=np.int64
    """
    start = np.array([1])
    end = 20
    step = 3
    dtype = np.int64
    res = np.arange(start, end, step).astype(dtype)
    obj.run(res=res, start=start, end=end, step=step, dtype=dtype)


@pytest.mark.api_base_arange_parameters
def test_arange5():
    """
    (np)start=1, (np)end=20, (np)step=3, dtype=np.float32
    """
    start = np.array([1])
    end = np.array([20])
    step = np.array([3])
    dtype = np.float32
    res = np.arange(start, end, step).astype(dtype)
    obj.run(res=res, start=start, end=end, step=step, dtype=dtype)


@pytest.mark.api_base_arange_parameters
def test_arange6():
    """
    start=1, end=20, (np)step=3, dtype=np.float32
    """
    start = 1
    end = 20
    step = np.array([3])
    dtype = np.float32
    res = np.arange(start, end, step).astype(dtype)
    obj.run(res=res, start=start, end=end, step=step, dtype=dtype)
