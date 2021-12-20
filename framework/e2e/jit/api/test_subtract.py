#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test subtract
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_subtract_vartype
def test_jit_subtract_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        return paddle.subtract(inputs, inputs_)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func,
        name="add_base",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps)
    runner.run()


@pytest.mark.jit_subtract_vartype
def test_jit_subtract_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        a = paddle.subtract(inputs, inputs_)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func,
        name="subtract_1",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps)
    runner.run()


@pytest.mark.jit_subtract_parameters
def test_jit_subtract_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        return paddle.subtract(inputs, inputs_)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="subtract_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps)
    runner.run()


@pytest.mark.jit_subtract_parameters
def test_jit_subtract_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        return paddle.subtract(inputs, inputs_)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    inps_ = randtool("float", -2, 2, shape=[2, 1, 1, 5, 1, 2])
    runner = Runner(func=func, name="subtract_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_subtract_parameters
def test_jit_subtract_4():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        return paddle.subtract(inputs, inputs_)

    inps = np.array([2, np.nan, 5])
    inps_ = np.array([1, 4, np.nan])
    runner = Runner(func=func, name="subtract_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_subtract_parameters
def test_jit_subtract_5():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.subtract(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.subtract
        """
        return paddle.subtract(inputs, inputs_)

    inps = np.array([5, np.inf, -np.inf])
    inps_ = np.array([1, 4, 5])
    runner = Runner(func=func, name="subtract_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()
