#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test imag
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_imag_vartype
def test_jit_imag_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.imag(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.imag
        """
        return paddle.imag(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="imag_base", dtype=["complex64", "complex128"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_imag_vartype
def test_jit_imag_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.imag(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.imag
        """
        a = paddle.imag(inputs)
        return a

    inps = np.array([1.5 + 3.0j, 2.1 - 2.0j, 3.2])
    runner = Runner(func=func, name="imag_1", dtype=["complex64", "complex128"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_imag_parameters
def test_jit_imag_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.imag(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.imag
        """
        return paddle.imag(inputs)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2]) + 3j
    runner = Runner(func=func, name="imag_2", dtype=["complex64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
