#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test add
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_minimum_vartype
def test_jit_minimum_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.minimum(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.minimum
        """
        return paddle.minimum(inputs, inputs_)

    inps = np.array([1.5, 2.1, 3.2])
    inps_ = np.array([1.5, 2.3, 5.2])
    runner = Runner(
        func=func,
        name="minimum_base",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_minimum_vartype
def test_jit_minimum_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.minimum(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.minimum
        """
        a = paddle.minimum(inputs, inputs_)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    inps_ = np.array([1.5, 2.3, 5.2])
    runner = Runner(
        func=func,
        name="minimum_1",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_minimum_parameters
def test_jit_minimum_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.minimum(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.minimum
        """
        return paddle.minimum(inputs, inputs_)

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    inps_ = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2]) + 1
    runner = Runner(func=func, name="minimum_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()
