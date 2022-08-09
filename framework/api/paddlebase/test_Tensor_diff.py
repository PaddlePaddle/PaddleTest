#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_diff
"""

import sys
import os
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2


class TestDiff(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64, np.int32, np.int64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-3


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestDiff(paddle.Tensor.diff)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_vartype
def test_diff_base():
    """
    base
    """
    x = np.random.randint(-4, 4, (4,))
    res = np.diff(x)
    obj.base(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff0():
    """
    default
    """
    x = np.random.rand(100)
    res = np.diff(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff1():
    """
    x: 2-d tensor
    """
    x = np.random.rand(4, 4)
    res = np.diff(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff2():
    """
    x: 3-d tensor
    """
    x = np.random.rand(4, 4, 4)
    res = np.diff(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff3():
    """
    x: 4-d tensor
    """
    x = np.random.rand(4, 4, 4, 4)
    res = np.diff(x)
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff4():
    """
    x: 4-d tensor
    axis=2
    """
    x = np.random.rand(4, 4, 4, 4)
    res = np.diff(x, axis=2)
    obj.run(res=res, x=x, axis=2)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff5():
    """
    x: 3-d tensor
    axis=-2
    """
    x = np.random.rand(4, 4, 4, 4)
    res = np.diff(x, axis=-2)
    obj.run(res=res, x=x, axis=-2)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff6():
    """
    x: 1d tensor
    set prepend
    """
    x = np.random.rand(10)
    b = np.random.rand(4)
    res = np.diff(x, prepend=b)
    obj.run(res=res, x=x, prepend=b)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff7():
    """
    x: 1d tensor
    set prepend
    set append
    """
    x = np.random.rand(10)
    b = np.random.rand(4)
    c = np.random.rand(4)
    res = np.diff(x, prepend=b, append=c)
    obj.run(res=res, x=x, prepend=b, append=c)


@skip_branch_is_2_2
@pytest.mark.api_base_diff_parameters
def test_diff8():
    """
    x: 2d tensor
    set prepend
    set append
    axis=0
    """
    x = np.random.rand(10, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)
    res = np.diff(x, axis=0, prepend=b, append=c)
    obj.run(res=res, x=x, axis=0, prepend=b, append=c)
