#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test real
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_real_vartype
def test_jit_real_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.real(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.real
        """
        return paddle.real(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="real_base", dtype=["complex64", "complex128"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_real_vartype
def test_jit_real_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.real(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.real
        """
        a = paddle.real(inputs)
        return a

    inps = np.array([1.5 + 3.0j, 2.1 - 2.0j, 3.2])
    runner = Runner(func=func, name="real_1", dtype=["complex64", "complex128"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_real_parameters
def test_jit_real_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.real(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.real
        """
        return paddle.real(inputs)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2]) + 3j
    runner = Runner(func=func, name="real_2", dtype=["complex64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
