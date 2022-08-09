#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sqrt
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_sqrt_vartype
def test_jit_sqrt_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sqrt(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sqrt
        """
        return paddle.sqrt(inputs)

    inps = np.array([1.5, 12.1, 3.2, 0])
    runner = Runner(func=func, name="sqrt_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_sqrt_vartype
def test_jit_sqrt_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sqrt(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sqrt
        """
        a = paddle.sqrt(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2, 0])
    runner = Runner(func=func, name="sqrt_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_sqrt_parameters
def test_jit_sqrt_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.sqrt(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.sqrt
        """
        return paddle.sqrt(inputs)

    inps = randtool("float", 0, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="sqrt_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
