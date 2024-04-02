#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test median
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_median_vartype
def test_jit_median_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        return paddle.median(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="median_base", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_median_vartype
def test_jit_median_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        a = paddle.median(inputs, axis=0)
        return a

    inps = np.array([[1.5, 2.1, 3.2], [1.5, 6.1, 3.4]])
    runner = Runner(func=func, name="median_1", dtype=["float32", "float64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_median_parameters
def test_jit_median_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        return paddle.median(inputs, axis=-3)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="median_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_median_parameters
def test_jit_median_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        return paddle.median(inputs, axis=-1, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="median_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_median_parameters
def test_jit_median_4():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        return paddle.median(inputs, axis=3, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="median_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_median_parameters
def test_jit_median_5():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.median(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.median
        """
        return paddle.median(inputs, axis=-2, keepdim=True)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1])
    runner = Runner(func=func, name="median_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
