#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test increment
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_increment_vartype
def test_jit_increment_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.increment(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.increment
        """
        return paddle.increment(inputs)

    inps = np.array([2.1])
    runner = Runner(func=func, name="increment_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_increment_vartype
def test_jit_increment_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.increment(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.increment
        """
        a = paddle.increment(inputs, value=2.1)
        return a

    inps = np.array([1.5])
    runner = Runner(func=func, name="increment_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_increment_parameters
def test_jit_increment_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.increment(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.increment
        """
        return paddle.increment(inputs, value=4.213)

    inps = np.array([3.0])
    runner = Runner(func=func, name="increment_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
