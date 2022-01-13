#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test shape
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_shape_vartype
def test_jit_shape_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.shape(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.shape
        """
        return paddle.shape(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="shape_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_shape_parameters
def test_jit_shape_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.shape(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.shape
        """
        return paddle.shape(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="shape_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_shape_parameters
def test_jit_shape_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.shape(inputs)
    inputs=[]
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.shape
        """
        return paddle.shape(inputs)

    inps = np.array([])
    runner = Runner(func=func, name="shape_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_shape_parameters
def test_jit_shape_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.shape(inputs)
    inputs=np.array([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.shape
        """
        return paddle.shape(inputs)

    inps = np.array([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
    runner = Runner(func=func, name="shape_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
