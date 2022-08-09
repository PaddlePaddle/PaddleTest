#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test tan
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_tan_vartype
def test_jit_tan_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.tan(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.tan
        """
        return paddle.tan(inputs)

    inps = np.array([1.5, -2.1, 3.2, 0])
    runner = Runner(func=func, name="tan_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_tan_vartype
def test_jit_tan_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.tan(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.tan
        """
        a = paddle.tan(inputs)
        return a

    inps = np.array([1.5, -2.1, 3.2, 0])
    runner = Runner(func=func, name="tan_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_tan_parameters
def test_jit_tan_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.tan(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.tan
        """
        return paddle.tan(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="tan_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
