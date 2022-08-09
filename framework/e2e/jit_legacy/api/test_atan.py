#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test atan
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_atan_vartype
def test_jit_atan_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.atan(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.atan
        """
        return paddle.atan(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="atan_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_atan_vartype
def test_jit_atan_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.atan(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.atan
        """
        a = paddle.atan(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="atan_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_atan_parameters
def test_jit_atan_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.atan(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.atan
        """
        return paddle.atan(inputs)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="atan_2", dtype=["float32"], ftype="func")
    runner.delta = 1e-6
    runner.rtol = 1e-7
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
