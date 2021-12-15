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


@pytest.mark.jit_bitwise_and_vartype
def test_jit_bitwise_and_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.bitwise_and(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.bitwise_and
        """
        return paddle.bitwise_and(inputs, inputs_)

    inps = np.array([1, 2, 3])
    inps_ = np.array([1, 2, 5])
    runner = Runner(
        func=func,
        name="add_base",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_bitwise_and_vartype
def test_jit_bitwise_and_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.bitwise_and(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.bitwise_and
        """
        a = paddle.bitwise_and(inputs, inputs_)
        return a

    inps = np.array([1, 2, 3])
    inps_ = np.array([1, 2, 5])
    runner = Runner(
        func=func,
        name="add_1",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()


@pytest.mark.jit_bitwise_and_parameters
def test_jit_bitwise_and_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.bitwise_and(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["int32"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.bitwise_and
        """
        return paddle.bitwise_and(inputs, inputs_)

    inps = randtool("int", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="add_2", dtype=["int32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps)
    runner.run()


@pytest.mark.jit_bitwise_and_parameters
def test_jit_bitwise_and_3():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.bitwise_and(inputs, inputs_)
    inps = np.array([1.5])
    inps_ = np.array([1.5, 2.3, 5.2])
    dtype=["bool"]
    """

    @paddle.jit.to_static
    def func(inputs, inputs_):
        """
        paddle.bitwise_and
        """
        a = paddle.bitwise_and(inputs, inputs_)
        return a

    inps = np.array([1])
    inps_ = np.array([1, 2, 5])
    runner = Runner(
        func=func,
        name="add_1",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["int32"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps, inputs_=inps_)
    runner.run()
