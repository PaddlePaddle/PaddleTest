#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test asin
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_asin_vartype
def test_jit_asin_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.asin(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.asin
        """
        return paddle.asin(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="asin_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_asin_vartype
def test_jit_asin_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.asin(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.asin
        """
        a = paddle.asin(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="asin_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_asin_parameters
def test_jit_asin_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.asin(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.asin
        """
        return paddle.asin(inputs)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="asin_2", dtype=["float32"], ftype="func")
    runner.delta = 1e-6
    runner.rtol = 1e-7
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
