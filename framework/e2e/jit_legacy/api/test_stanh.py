#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test stanh
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_stanh_vartype
def test_jit_stanh_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.stanh(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.stanh
        """
        return paddle.stanh(inputs, scale_a=0.33, scale_b=-1.7159)

    inps = np.array([1.5, -2.1, 3.2, 0])
    runner = Runner(func=func, name="stanh_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_stanh_vartype
def test_jit_stanh_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.stanh(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.stanh
        """
        a = paddle.stanh(inputs, scale_a=0.1, scale_b=1.2)
        return a

    inps = np.array([1.5, -2.1, 3.2, 0])
    runner = Runner(func=func, name="stanh_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_stanh_parameters
def test_jit_stanh_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.stanh(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.stanh
        """
        return paddle.stanh(inputs, scale_a=0.67, scale_b=1.7159)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="stanh_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
