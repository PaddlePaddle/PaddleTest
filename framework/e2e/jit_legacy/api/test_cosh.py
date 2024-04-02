#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cosh
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_cosh_vartype
def test_jit_cosh_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cosh(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cosh
        """
        return paddle.cosh(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="cosh_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_cosh_parameters
def test_jit_cosh_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cosh(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cosh
        """
        return paddle.cosh(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="cosh_1", dtype=["float32"], ftype="func")
    runner.delta = 1e-6
    runner.rtol = 1e-7
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
    runner.delta = 1e-10
    runner.rtol = 1e-11


@pytest.mark.jit_cosh_parameters
def test_jit_cosh_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.cosh(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float16"]
    only support gpu
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.cosh
        """
        return paddle.cosh(inputs)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="cosh_2", dtype=["float16"], ftype="func")
    runner.places = ["gpu:0"]
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    if paddle.device.is_compiled_with_cuda() is True:
        runner.run()
