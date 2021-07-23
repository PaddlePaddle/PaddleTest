#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test meshgrid
"""
from apibase import randtool
from apibase import compare
import paddle
import pytest
from paddle import fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np


if fluid.is_compiled_with_cuda() is True:
    places = [fluid.CPUPlace(), fluid.CUDAPlace(0)]
else:
    places = [fluid.CPUPlace()]


@pytest.mark.api_base_meshgrid_vartype
def test_meshgrid_base():
    """
    base
    """
    types = [np.int32, np.int64, np.float32, np.float64]
    for t in types:
        x = randtool("float", -10, 10, [2]).astype(t)
        y = randtool("float", -10, 10, [4]).astype(t)
        for place in places:
            paddle.disable_static(place)
            res = paddle.meshgrid(to_variable(x), to_variable(y))
            expect = np.meshgrid(x, y)
            for i, exp in enumerate(expect):
                expect[i] = exp.transpose(1, 0)
            compare(res, expect)
            paddle.enable_static()


@pytest.mark.api_base_meshgrid_parameters
def test_meshgrid():
    """
    default input=x,y,z
    """
    types = [np.int32, np.int64, np.float32, np.float64]
    for t in types:
        x = randtool("float", -10, 10, [2]).astype(t)
        y = randtool("float", -10, 10, [4]).astype(t)
        z = randtool("float", -10, 10, [6]).astype(t)
        for place in places:
            paddle.disable_static(place)
            res = paddle.meshgrid(to_variable(x), to_variable(y), to_variable(z))
            expect = np.meshgrid(x, y, z)
            for i, exp in enumerate(expect):
                expect[i] = exp.transpose(1, 0, 2)
            compare(res, expect)
            paddle.enable_static()
