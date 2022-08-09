#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lgamma
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_lgamma_vartype
def test_jit_lgamma_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.lgamma(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.lgamma
        """
        return paddle.lgamma(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="lgamma_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_lgamma_vartype
def test_jit_lgamma_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.lgamma(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.lgamma
        """
        a = paddle.lgamma(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="lgamma_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_lgamma_parameters
def test_jit_lgamma_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.lgamma(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.lgamma
        """
        return paddle.lgamma(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="lgamma_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
