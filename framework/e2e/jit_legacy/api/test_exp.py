#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test exp
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_exp_vartype
def test_jit_exp_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.exp(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.exp
        """
        return paddle.exp(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="exp_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_exp_vartype
def test_jit_exp_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.exp(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.exp
        """
        a = paddle.exp(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="exp_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_exp_parameters
def test_jit_exp_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.exp(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.exp
        """
        return paddle.exp(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="exp_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
