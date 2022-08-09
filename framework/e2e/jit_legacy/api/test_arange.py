#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test arange
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_arange_vartype
def test_jit_arange_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.arange(inputs, end=5, step=1)
    inps = np.array([3])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.arange
        """
        return paddle.arange(inputs, end=5, step=1)

    inps = np.array([3])
    runner = Runner(func=func, name="add_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_arange_vartype
def test_jit_arange_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.arange(inputs)
    inps = np.array([2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.arange
        """
        a = paddle.arange(inputs)
        return a

    inps = np.array([2])
    runner = Runner(func=func, name="arange_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_arange_vartype
def test_jit_arange_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        a = paddle.arange(inputs, end=102, step=3)
        return a
    inps = np.array([2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.arange
        """
        a = paddle.arange(inputs, end=102, step=3)
        return a

    inps = np.array([2])
    runner = Runner(func=func, name="arange_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
