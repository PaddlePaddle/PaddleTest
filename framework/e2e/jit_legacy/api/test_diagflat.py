#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diagflat
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_diagflat_vartype
def test_jit_diagflat_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagflat(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagflat
        """
        return paddle.diagflat(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="diagflat_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diagflat_vartype
def test_jit_diagflat_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagflat(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagflat
        """
        a = paddle.diagflat(inputs)
        return a

    inps = np.array([[1.5, 2.1, 3.2], [-2.5, -1.1, 0]])
    runner = Runner(func=func, name="diagflat_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diagflat_parameters
def test_jit_diagflat_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagflat(inputs)
    inputs=paddle.rand([10, 6])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagflat
        """
        return paddle.diagflat(inputs, offset=3)

    inps = randtool("float", -2, 2, shape=[10, 6])
    runner = Runner(func=func, name="diagflat_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diagflat_parameters
def test_jit_diagflat_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagflat(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagflat
        """
        return paddle.diagflat(inputs, offset=-2)

    inps = randtool("float", -2, 2, shape=[5, 6, 5])
    runner = Runner(func=func, name="diagflat_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
