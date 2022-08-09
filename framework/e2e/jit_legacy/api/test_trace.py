#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test trace
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_trace_vartype
def test_jit_trace_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.trace(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.trace
        """
        return paddle.trace(inputs, offset=0, axis1=0, axis2=1)

    inps = randtool("float", -2, 2, shape=[3, 6])
    runner = Runner(func=func, name="trace_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_trace_vartype
def test_jit_trace_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.trace(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.trace
        """
        a = paddle.trace(inputs, offset=1, axis1=1, axis2=-2)
        return a

    inps = randtool("float", -2, 2, shape=[3, 6])
    runner = Runner(func=func, name="trace_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_trace_parameters
def test_jit_trace_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.trace(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.trace
        """
        return paddle.trace(inputs, offset=-1, axis1=2, axis2=4)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="trace_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
