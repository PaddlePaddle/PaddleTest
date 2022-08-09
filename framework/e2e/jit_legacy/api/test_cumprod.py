#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cumprod
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_cumprod_vartype
def test_jit_cumprod_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cumprod(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cumprod
        """
        return paddle.cumprod(inputs, dim=0)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="cumprod_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_cumprod_parameters
def test_jit_cumprod_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cumprod(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    dim=0
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cumprod
        """
        return paddle.cumprod(inputs, dim=0)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="cumprod_1", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_cumprod_parameters
def test_jit_cumprod_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cumprod(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    dim=-1
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cumprod
        """
        return paddle.cumprod(inputs, dim=-1)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="cumprod_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_cumprod_parameters
def test_jit_cumprod_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cumprod(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    dim=-3
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cumprod
        """
        return paddle.cumprod(inputs, dim=-3)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="cumprod_3", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
