#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test randperm
"""
import sys
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np

sys.path.append("../..")
from utils.interceptor import skip_platform_not_linux, skip_not_compile_gpu


class TestRandperm(APIBase):
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


obj = TestRandperm(paddle.randperm)


@skip_platform_not_linux
@pytest.mark.api_base_randperm_vartype
def test_randperm_base():
    """
    base
    """
    obj.places = [paddle.CPUPlace()]
    res = np.array([0, 1, 4, 3, 2, 5])
    n = 6
    obj.base(res=res, n=n)


@skip_not_compile_gpu
@skip_platform_not_linux
@pytest.mark.api_base_randperm_parameters
def test_randperm():
    """
    default
    """
    obj.places = [paddle.CUDAPlace(0)]
    res = np.array([9, 4, 0, 7, 1, 5, 2, 3, 6, 8])
    n = 10
    obj.run(res=res, n=n)


@skip_platform_not_linux
@pytest.mark.api_base_randperm_parameters
def test_randperm1():
    """
    seed = 1
    """
    obj.places = [paddle.CPUPlace()]
    obj.seed = 1
    res = np.array([6, 3, 7, 8, 9, 2, 1, 5, 4, 0])
    n = 10
    obj.run(res=res, n=n)


@skip_not_compile_gpu
@skip_platform_not_linux
@pytest.mark.api_base_randperm_parameters
def test_randperm2():
    """
    dtype = np.float32
    """
    obj.places = [paddle.CUDAPlace(0)]
    obj.seed = 33
    res = np.array([9.0, 4.0, 0.0, 7.0, 1.0, 5.0, 2.0, 3.0, 6.0, 8.0])
    n = 10
    obj.run(res=res, n=n, dtype=np.float32)


@skip_platform_not_linux
@pytest.mark.api_base_randperm_exception
def test_randperm3():
    """
    exception n < 0 BUG
    """
    obj.seed = 33
    # res = np.array([0.0, 1.0, 6.0, 2.0, 9.0, 3.0, 5.0, 7.0, 4.0, 8.0])
    n = -1
    obj.exception(etype="InvalidArgument", n=n, dtype=np.float32)


@skip_platform_not_linux
@pytest.mark.api_base_randperm_exception
def test_randperm4():
    """
    exception dtype = np.int8 BUG
    """
    obj.seed = 33
    # res = np.array([0.0, 1.0, 6.0, 2.0, 9.0, 3.0, 5.0, 7.0, 4.0, 8.0])
    n = -1
    obj.exception(etype="NotFound", n=n, dtype=np.int8)
