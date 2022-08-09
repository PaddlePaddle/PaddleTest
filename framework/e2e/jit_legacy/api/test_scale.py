#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test scale
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_scale_vartype
def test_jit_scale_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        return paddle.scale(inputs, scale=1.0, bias=0.0, bias_after_scale=True, act=None)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="scale_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_scale_vartype
def test_jit_scale_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        a = paddle.scale(inputs, scale=2.0, bias=-1.0, bias_after_scale=False, act=None)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="scale_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_scale_parameters
def test_jit_scale_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        return paddle.scale(inputs, scale=1.5, bias=-100.0, bias_after_scale=True, act="relu")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="scale_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_scale_parameters
def test_jit_scale_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        return paddle.scale(inputs, scale=1.5, bias=-2.0, bias_after_scale=True, act="softmax")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="scale_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_scale_parameters
def test_jit_scale_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        return paddle.scale(inputs, scale=1.5, bias=-2.0, bias_after_scale=False, act="tanh")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="scale_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_scale_parameters
def test_jit_scale_4():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.scale(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.scale
        """
        return paddle.scale(inputs, scale=1.5, bias=-2.0, bias_after_scale=False, act="tanh")

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="scale_2", dtype=["float32"], ftype="sigmoid")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
