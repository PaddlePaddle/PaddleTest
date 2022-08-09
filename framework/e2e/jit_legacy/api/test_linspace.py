#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test linspace
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_linspace_vartype
def test_jit_linspace_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.linspace(inputs, stop=11, num=3)
    inps = np.array([-5])
    dtype=["float64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.linspace
        """
        return paddle.linspace(inputs, inputs_, num=3)

    inps = np.array([-5])
    inps_ = np.array([10])
    runner = Runner(func=func, name="linspace_base", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_linspace_vartype
def test_jit_linspace_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        a = paddle.linspace(inputs, stop=10, num=4)
        return a
    inps = np.array([0])
    dtype=["int32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.linspace
        """
        a = paddle.linspace(inputs, stop=10, num=4)
        return a

    inps = np.array([0])
    runner = Runner(func=func, name="linspace_1", dtype=["int32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_linspace_parameters
def test_jit_linspace_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.linspace(inputs)
    inps = np.array([3])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.linspace
        """
        return paddle.linspace(inputs, stop=5, num=21)

    inps = np.array([3])
    runner = Runner(func=func, name="linspace_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_linspace_vartype
def test_jit_linspace_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        a = paddle.linspace(inputs, stop=10, num=4)
        return a
    inps = np.array([0])
    dtype=["int32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.linspace
        """
        a = paddle.linspace(inputs, stop=10, num=4)
        return a

    inps = np.array([0])
    runner = Runner(func=func, name="linspace_1", dtype=["int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
