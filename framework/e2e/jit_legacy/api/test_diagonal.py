#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diagonal
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_diagonal_vartype
def test_jit_diagonal_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagonal(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagonal
        """
        return paddle.diagonal(inputs)

    inps = np.array([[1.5, 2.1, 3.2], [-1.1, -3.3, 7.2]])
    runner = Runner(
        func=func,
        name="diagonal_base",
        # dtype=["float32", "float64", "float16"],
        dtype=["int32", "int64", "float32", "float64"],
        ftype="func",
    )
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diagonal_parameters
def test_jit_diagonal_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagonal(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagonal
        """
        return paddle.diagonal(inputs, offset=1, axis1=2, axis2=-1)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="diagonal_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diagonal_parameters
def test_jit_diagonal_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diagonal(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float16"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diagonal
        """
        return paddle.diagonal(inputs, offset=2, axis1=3, axis2=-2)

    inps = randtool("float", -1, 1, shape=[3, 6, 2, 2, 2, 1, 5, 4, 2])
    runner = Runner(func=func, name="diagonal_2", dtype=["float16"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    if paddle.device.is_compiled_with_cuda() is True:
        runner.run()
