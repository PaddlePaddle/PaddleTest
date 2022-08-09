#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test rsqrt
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_rsqrt_vartype
def test_jit_rsqrt_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.rsqrt(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.rsqrt
        """
        return paddle.rsqrt(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="rsqrt_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_rsqrt_vartype
def test_jit_rsqrt_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.rsqrt(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.rsqrt
        """
        a = paddle.rsqrt(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="rsqrt_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_rsqrt_parameters
def test_jit_rsqrt_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.rsqrt(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.rsqrt
        """
        return paddle.rsqrt(inputs)

    inps = randtool("float", 1, 4, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="rsqrt_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
