#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test diag
"""
import pytest
import paddle
import numpy as np
from jitbase import Runner
from jitbase import randtool


@pytest.mark.jit_diag_vartype
def test_jit_diag_base():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diag(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diag
        """
        return paddle.diag(inputs)

    inps = np.array([1.5, 2.1, 3.2])
    runner = Runner(func=func, name="diag_base", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diag_vartype
def test_jit_diag_1():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diag(inputs)
    inputs=np.array([1.5, 2.1, 3.2])
    dtype=["float32", "float64", "int32", "int64"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diag
        """
        a = paddle.diag(inputs)
        return a

    inps = np.array([[1.5, 2.1, 3.2], [-2.5, -1.1, 0]])
    runner = Runner(func=func, name="diag_1", dtype=["float32", "float64", "int32", "int64"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()


@pytest.mark.jit_diag_parameters
def test_jit_diag_2():
    """
    @paddle.jit.to_static
    def fun(inputs):
        return paddle.diag(inputs)
    inputs=paddle.rand([3, 6, 2, 2, 2, 1, 5, 4, 2])
    dtype=["float32"]
    """

    @paddle.jit.to_static
    def func(inputs):
        """
        paddle.diag
        """
        return paddle.diag(inputs, offset=3, padding_value=2.1)

    inps = randtool("float", -2, 2, shape=[10, 6])
    runner = Runner(func=func, name="diag_2", dtype=["float32"], ftype="func")
    runner.add_kwargs_to_dict("params_group1", inputs=inps)
    runner.run()
