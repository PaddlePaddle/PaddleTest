#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test expm1
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_expm1_vartype
def test_jit_expm1_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.expm1(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.expm1
        """
        return paddle.expm1(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="expm1_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_expm1_vartype
def test_jit_expm1_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.expm1(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.expm1
        """
        a = paddle.expm1(inputs)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="expm1_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_expm1_parameters
def test_jit_expm1_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.expm1(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.expm1
        """
        return paddle.expm1(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="expm1_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_expm1_parameters
def test_jit_expm1_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.expm1(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.expm1
        """
        return paddle.expm1(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="expm1_3", dtype=["float16"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    if paddle.device.is_compiled_with_cuda() is True:
        runner.run()
