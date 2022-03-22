#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_Tensor_repeat_interleave
"""

from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestRepeatInterleave(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = False
        # enable check grad
        # self.enable_backward = False
        # self.delta = 1e-5


obj = TestRepeatInterleave(paddle.Tensor.repeat_interleave)


@pytest.mark.api_base_repeat_interleave_vartype
def test_repeat_interleave_base():
    """
    base
    """
    x = randtool("float", 0, 1, (4,))
    res = np.repeat(x, 3)
    obj.base(res=res, x=x, repeats=3)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave0():
    """
    x: 2d-tensor
    """
    x = randtool("float", -2, 2, (4, 2))
    res = np.repeat(x, 2)
    obj.run(res=res, x=x, repeats=2)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave1():
    """
    x: 3d-tensor
    """
    x = randtool("float", -2, 2, (4, 2, 4))
    res = np.repeat(x, 2)
    obj.run(res=res, x=x, repeats=2)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave2():
    """
    x: 4d-tensor
    """
    x = randtool("float", -2, 2, (4, 2, 4, 5))
    res = np.repeat(x, 2)
    obj.run(res=res, x=x, repeats=2)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave3():
    """
    x: 5d-tensor
    """
    x = randtool("float", -2, 2, (4, 2, 4, 4, 5))
    res = np.repeat(x, 2)
    obj.run(res=res, x=x, repeats=2)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave4():
    """
    x: 5d-tensor
    axis = 1
    """
    x = randtool("float", -2, 2, (4, 2, 4, 4, 5))
    res = np.repeat(x, 2, axis=1)
    obj.run(res=res, x=x, repeats=2, axis=1)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave5():
    """
    x: 5d-tensor
    axis = 3
    type: int
    """
    obj.types = [np.int32, np.int64]
    x = randtool("int", -2, 2, (4, 2, 4, 4, 5))
    res = np.repeat(x, 2, axis=3)
    obj.run(res=res, x=x, repeats=2, axis=3)


@pytest.mark.api_base_repeat_interleave_parameters
def test_repeat_interleave6():
    """
    x: 5d-tensor
    axis = 2
    type: int
    repeats:Tensor
    """
    obj1 = TestRepeatInterleave(paddle.repeat_interleave)
    obj1.types = [np.int32, np.int64]
    obj1.enable_backward = False
    x = randtool("int", -2, 2, (4, 2, 4, 4, 5))
    repeat = np.array([2, 4], dtype=np.int64)
    res = np.repeat(x, repeat, axis=1)
    obj1.run(res=res, x=x, repeats=repeat, axis=1)
