#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test pow
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_pow_vartype
def test_jit_pow_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.pow(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.pow
        """
        return paddle.pow(inputs, 3)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func,
        name="add_base",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_pow_vartype
def test_jit_pow_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.pow(inputs, inputs_)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.pow
        """
        a = paddle.pow(inputs, 2.3)
        return a

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(
        func=func,
        name="pow_1",
        # dtype=["float32", "float64", "int32", "int64", "float16"],
        dtype=["float32", "float64", "int32", "int64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_pow_parameters
def test_jit_pow_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.pow(inputs, inputs_)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.pow
        """
        return paddle.pow(inputs, paddle.to_tensor([3.0], dtype="float32"))

    inps = randtool("float", -2, 2, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="pow_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
