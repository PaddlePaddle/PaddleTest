#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_rot90
"""

import sys
import os
from apibase import APIBase
import paddle
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_branch_is_2_2


class TestRot90(APIBase):
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
        # self.enable_backward = False
        # self.delta = 1e-3


if os.getenv("AGILE_COMPILE_BRANCH") != "release/2.2":
    obj = TestRot90(paddle.rot90)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_vartype
def test_rot_base():
    """
    base
    """
    x = np.random.randint(-4, 4, (4, 4))
    res = np.rot90(x, k=1, axes=[0, 1])
    obj.base(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot0():
    """
    default
    """
    x = np.random.randint(-4, 4, (3, 4))
    res = np.rot90(x, k=1, axes=[0, 1])
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot1():
    """
    x: 3-d
    """
    x = np.random.randint(-4, 4, (4, 4, 4))
    res = np.rot90(x, k=1, axes=[0, 1])
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot2():
    """
    x: 4-d
    """
    x = np.random.randint(-4, 4, (4, 4, 4, 4))
    res = np.rot90(x, k=1, axes=[0, 1])
    obj.run(res=res, x=x)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot3():
    """
    x: 2-d
    k = 4
    """
    x = np.random.randint(-4, 4, (4, 4))
    res = np.rot90(x, k=4, axes=[0, 1])
    obj.run(res=res, x=x, k=4)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot4():
    """
    x: 2-d
    k = -1
    """
    x = np.random.randint(-4, 4, (4, 4))
    res = np.rot90(x, k=-1, axes=[0, 1])
    obj.run(res=res, x=x, k=-1)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot5():
    """
    x: 2-d
    k = -4
    """
    x = np.random.randint(-4, 4, (2, 2))
    res = np.rot90(x, k=-4, axes=[0, 1])
    obj.run(res=res, x=x, k=-4)


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot6():
    """
    x: 4-d
    k = -1
    axes = [1, 2]
    """
    x = np.random.randint(-4, 4, (4, 4, 4, 4))
    res = np.rot90(x, k=-1, axes=[1, 2])
    obj.run(res=res, x=x, k=-1, axes=[1, 2])


@skip_branch_is_2_2
@pytest.mark.api_base_rot90_parameters
def test_rot7():
    """
    x: 4-d
    k = -1
    axes = (2, 3)
    """
    x = np.random.randint(-4, 4, (4, 4, 4, 4))
    res = np.rot90(x, k=-1, axes=(2, 3))
    obj.run(res=res, x=x, k=-1, axes=(2, 3))
